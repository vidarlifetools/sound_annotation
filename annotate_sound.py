#!/usr/bin/env python3
from tkinter import *
from tkinter.ttk import *
import os
from os import listdir
import numpy as np
#from PIL import ImageTk, Image
#import cv2
from google.cloud import storage
import argparse
import sys
from os.path import isfile, join
from datetime import datetime
import json
import subprocess
from functools import partial
from pathlib import Path
#from PIL import Image, ImageTk

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

sys.path.append("..")

GSTORAGE_JSON = "/home/vidar/projects/knowmeai/sensors/KnowMeAI-326518239433.json"
BUCKET = "knowmeai_bucket"


class Gui(Tk):
    def __init__(self, gfiles, folder, gclient):
        Tk.__init__(self)

        self.gclient = gclient
        self.gfiles = gfiles
        self.target_folder = folder
        self.bucket = gclient.bucket(BUCKET)
        self.annotations = {}


        # Clients menu
        clients = sorted(
            list(
                set("/".join(i.name.split("/")[:-1]) for i in gfiles if "raw" in i.name)
            )
        )
        clients.append(clients[0])
        self.clients_var = StringVar()
        self.clients_menu = OptionMenu(
            self, self.clients_var, *clients, command=self.update_json_list
        )
        self.clients_menu.pack(side=TOP, anchor=NW, padx=10, pady=10)

        # Json file list
        self.file_list = Treeview(
            #self, columns=("file", "annotation_status", "gitem", "actual_bbox")
            self, columns = ("file", "annotation_status", "video")
        )
        #self.file_list.heading("#1", text="file", anchor=W)
        self.file_list.heading("#1", text="File", anchor=W)
        self.file_list.heading("#2", text="Sound annotation", anchor=W)
        self.file_list.column("#1", stretch=YES, minwidth=300)
        self.file_list.column("#2", stretch=NO, minwidth=50)
        self.file_list.column("#3", stretch=NO, minwidth=0, width=0)
        self.file_list.bind("<Double-1>", self.annotate_files)

        scrollbar = Scrollbar(self, command=self.file_list.yview)
        scrollbar.pack(side=LEFT, fill=Y, anchor=NW, padx=10, pady=10)
        self.file_list.pack(fill=BOTH, expand=1, side=LEFT, padx=10, pady=10, anchor=W)
        #self.img_panel.pack(fill=BOTH, expand=1, side=LEFT, padx=10, pady=10, anchor=NW)

        self.bboxes = []
        self.img_scale = 0
        self.current_frame_num = -1
        self.update()
        self.geometry("500x760")

        self.mainloop()

    def upload_annotation_files(self, sound_file):
        base = join(self.target_folder, "annotation")
        files = listdir(base)
        for file in files:
            source_file = join(base, file)
            if isfile(source_file):
                dest_file = join("annotation", sound_file.split("/")[1], "annotation", file)
                print("copying from ", source_file, "  to  ", dest_file)
                blob = self.bucket.blob(dest_file)
                blob.upload_from_filename(source_file)
        os.system("rm " + base + "/*.json")

    def create_annotation_files(self, sound_file):
        label_lookup = {
            "cli": ("sound_client", 101),
            "car": ("sound_caretaker", 102),
            "env": ("sound_environment", 103)
        }
        base = join(self.target_folder, "annotation")
        if not isfile(join(base, "Label.txt")):
            print("Label.txt not found: ", join(base, "Label.txt"))
            return False
        with open(join(base, "Label.txt"), "r") as fp:
            text = fp.read()
            text = text.lower().split("\n")
            for annotation in text:
                annotation = annotation.split("\t")
                if len(annotation) == 3:
                    start_time = float(annotation[0])
                    stop_time = float(annotation[1])
                    label_name = label_lookup[annotation[2]][0]
                    label_id = label_lookup[annotation[2]][1]
                    print(start_time, stop_time, label_name, label_id)
                    # Create the annotation file
                    annotation_descr = {}
                    annotation_descr['video'] = join("https://storage.cloud.google.com",
                                                     BUCKET,
                                                     "annotation",
                                                     sound_file.split("/")[1],
                                                     "raw",
                                                     sound_file.split("/")[-1].replace(".wav", ".mp4"))
                    annotation_descr["label_name"] = label_name
                    annotation_descr["label_id"] = label_id
                    annotation_descr["start"] = start_time
                    annotation_descr["end"] = stop_time
                    annotation_descr["context"] = []
                    contexts = {
                        ("When", 1, "morning", 1),
                        ("Where", 2, "familiar", 1),
                        ("Position", 3, "sitting", 1)
                    }
                    for c in contexts:
                        temp = {}
                        temp["category_name"] = c[0]
                        temp["category_id"] = c[1]
                        temp["item_name"] = c[2]
                        temp["item_id"] = c[3]
                        annotation_descr["context"].append(temp)
                    annotation_descr["significance"] = []
                    annotation_descr["significance"].append("sound")

                    # Save the annotation description to the bucket
                    t = datetime.now()
                    annotation_filename = t.strftime("%Y-%m-%d-%H-%M-%S-%f")
                    dest_filename = join(base, annotation_filename + ".json")
                    with open(dest_filename, 'w') as fp:
                        json.dump(annotation_descr, fp, indent=4)
            print("Deleting ", join(base, "*.txt"))
            os.system("rm " + join(base, "*.txt"))
            return True

    def annotate_files(self, _, frame_num=0):
        f = self.file_list.focus()
        f_file, f_status, sound_file = self.file_list.item(f)["values"]

        # Create an import-file with current annotations to Audicity
        if sound_file in self.annotations.keys():
            with open("data/annotation/Import.txt", "w") as fp:
                for i in range(len(self.annotations[sound_file])):
                    fp.write(str(self.annotations[sound_file][i][0]) + "\t" +
                            str(self.annotations[sound_file][i][1]) + "\t" +
                            ["non", "cli", "car", "env"][self.annotations[sound_file][i][2]- 100] + "\n")
                    blob_name = self.annotations[sound_file][i][3]
                    print("Annotation file to be deleted: ", blob_name)
                    blob = self.bucket.blob(blob_name)
                    if blob.exists():
                        blob.delete()

        # Get file from google storage
        dest_file = join(self.target_folder, sound_file.split("/")[-1])
        sound_blob = self.bucket.blob(sound_file)
        sound_blob.download_to_filename(dest_file)
        # Start audacity
        p = subprocess.Popen(("/snap/bin/audacity", dest_file))
        p.wait()
        os.system("rm " + "data/annotation/Import.txt")
        os.system("rm " + dest_file)
        # Create annotation files
        if self.create_annotation_files(sound_file):
            self.file_list.insert("", 0, values=(f_file, "Yes", sound_file))
            self.file_list.delete(f)

        self.upload_annotation_files(sound_file)

    def get_annotation_status(self, json_file):
        blob = self.bucket.blob(json_file)
        annotation = json.loads(blob.download_as_string())
        sound_file = annotation.get("video", None).replace("https://storage.cloud.google.com/knowmeai_bucket/", "")
        sound_file = sound_file.replace(".mp4", ".wav")

        # update a list of sound annotations for the recording
        if annotation.get("label_id", None) >= 100:
            if sound_file not in self.annotations.keys():
                self.annotations[sound_file] = []
            self.annotations[sound_file].append([annotation.get("start", None),
                                                 annotation.get("end", None),
                                                 annotation.get("label_id", None),
                                                 json_file])
        return annotation.get("label_id", None) >= 100, sound_file


    def update_json_list(self, value):
        self.file_list.delete(*self.file_list.get_children())

        client_folder = self.clients_var.get()
        sound_files = [
            f.name
            for f in self.gfiles
            if client_folder in f.name
            and ".wav" in f.name
            and "_post" not in f.name
        ]

        annotation_folder = client_folder.replace("/raw", "/annotation")
        annotation_files = [
            f.name
            for f in self.gfiles
            if annotation_folder in f.name
               and ".json" in f.name
               and "xref" not in f.name

        ]
        # Mark raw files with sound annotation
        raw_list = {}
        for file in sound_files:
            raw_list[file] = False

        for file in annotation_files:
            annotated, sound_file = self.get_annotation_status(file)
            if sound_file in raw_list.keys() and annotated:
                raw_list[sound_file] = True
            else:
                raw_list[sound_file] = annotated


        for item in raw_list:
            fname = item.split("/")[-1]
            annotation_status = "Yes" if raw_list[item] else "No"
            self.file_list.insert("", 0, values=(fname, annotation_status, item))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="Folder for downloaded files", type=Path, default=str(Path.cwd()) + "/data/"
    )
    args = parser.parse_args()
    print("folder", args.folder)

    client = storage.Client.from_service_account_json(GSTORAGE_JSON)
    gfiles = [_ for _ in client.list_blobs(BUCKET)]
    Gui(gfiles, args.folder, client)
