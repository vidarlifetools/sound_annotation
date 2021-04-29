import argparse
import json
from os import listdir
from os.path import isfile, join
from datetime import datetime
#import cv2
import os
import subprocess
from google.cloud import storage

def get_files_to_annotate(config, storage_client):
    client = config["client"]
    base = config["destination_directory"]
    # Delete all present files
    if len(listdir(base)) > 0:
        print("Deleting old annotation files")
        os.system("rm " + join(base+"*.json"))
        os.system("rm " + join(base+"*.wav"))

    blobs = storage_client.list_blobs(config["google_storage_bucket"])
    bucket = storage_client.bucket(config["google_storage_bucket"])
    for blob in blobs:
        if "annotation/"+client+"/annotation/" in blob.name\
                and not "xref" in blob.name\
                and blob.name.split("/")[3] != "":
            print("Collecting: ", blob.name)
            dest_file = (
                    config["destination_directory"]
                    + str(blob.name.split("/")[-1])
            )
            context_blob = bucket.blob(blob.name)
            context_blob.download_to_filename(dest_file)
    # Make a list of raw files that has been sound annotated
    annotated_files = []
    annotation_files = [f for f in listdir(base) if isfile(join(base, f))]
    for annotation_file in annotation_files:
        annotation = json.load(open(base+annotation_file))
        if annotation["label_id"] >= 100:
            if not annotation["video"].split("/")[-1] in annotated_files:
                annotated_files.append(annotation["video"].split("/")[-1])
    raw_files = []
    # Make a list of raw files that has not been sound annotated
    blobs = storage_client.list_blobs(config["google_storage_bucket"])
    for blob in blobs:
        if "annotation/"+client+"/raw/" in blob.name\
                and not "inspected" in blob.name\
                and not "_depth" in blob.name\
                and "mp4" in blob.name\
                and blob.name.split("/")[3] != "":
            if blob.name.split("/")[-1] not in annotated_files:
                raw_files.append(blob.name.split("/")[-1])
    return raw_files

label_lookup = {
    "cli": ("sound_client", 101),
    "car": ("sound_caretaker", 102),
    "env": ("sound_environment", 103)
}

def create_annotation_files(config, raw_file):
    base = join(config["destination_directory"], "annotation")
    if not isfile(join(base, "Label Track.txt")):
        print("Label Track.txt not found")
        return
    with open(join(base, "Label Track.txt"), "r") as fp:
        text = fp.read()
        text = text.split("\n")
        print(text)
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
                                                 config["google_storage_bucket"],
                                                 "raw",
                                                 raw_file)
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
        os.remove(join(base, "Label Track.txt"))

def upload_annotation_files(config, storage_client):
    base = join(config["destination_directory"], "annotation")
    bucket = storage_client.bucket("knowmeai_bucket")
    files = listdir(base)
    for file in files:
        source_file = join(base, file)
        if isfile(source_file):
            dest_file = join("annotation", config["client"], "annotation", file)
            print("copying from ", source_file, "  to  ", dest_file )
            blob = bucket.blob(dest_file)
            blob.upload_from_filename(source_file)
    os.system("rm " + base + "/*.json")


def annotate_files(config, raw_files):
    client = config["client"]
    base = config["destination_directory"]
    bucket = storage_client.bucket(config["google_storage_bucket"])
    for i, raw_file in enumerate(raw_files):
        # Get file from google storage
        dest_file = base + str(raw_file.split(".")[0] + ".wav")
        context_blob = bucket.blob(join("annotation", client, "raw", raw_file.split(".")[0] + ".wav"))
        context_blob.download_to_filename(dest_file)
        # Start audacity
        p = subprocess.Popen(("/snap/bin/audacity", dest_file))
        p.wait()
        os.system("rm " + dest_file)
        # Create annotation files
        create_annotation_files(config, raw_file)


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config file", default="config.json")
args = parser.parse_args()

config = json.load(open(args.config))
storage_client = storage.Client.from_service_account_json(
    config["google_storage_key"]
)

raw_files = get_files_to_annotate(config, storage_client)
annotate_files(config, raw_files)
upload_annotation_files(config, storage_client)
exit()
