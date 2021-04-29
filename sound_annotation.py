from google.cloud import storage
import sys
import os
import time
import pickle
#from dataclasses import dataclass
import argparse
import json
from os import listdir
from os.path import isfile, join
from datetime import datetime
#import cv2
import os
import subprocess
import time
#import numpy as np
import pickle
#import soundfile as sf
from collections import deque
from google.cloud import storage
#import torch
#from depth_compression import decode_colorized
#from utils import load_video_frames


sys.path.append("..")
#from preprocessing.feature_detection import FeatureDetector

#from dataclasses import dataclass
#from preprocessing.sound_features import sound_feature

def get_files_to_annotate(config, storage_client):
    client = config["client"]
    base = config["destination_directory"]
    # Delete all present files
    if len(listdir(base)) > 0:
        print("Deleting old annotation files")
        os.system("rm " + join(base+"*.json"))
        if isfile(join(base+"*.wav")):
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
    "env": ("sound_environment", 102),
    "car": ("sound_caretaker", 102)
}

def create_annotation_files(config, raw_file):
    base = join(config["destination_directory"], "annotation")
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

def upload_annotation_files(config):
    storage_client = storage.Client()
    bucket = storage_client.bucket("knowmeai_bucket")
    #blob = bucket.blob(dest_filename)
    #blob.upload_from_filename("tmp.json")


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


#raw_files = get_files_to_annotate(config, storage_client)
#annotate_files(config, raw_files)
create_annotation_files(config, "2021-3-4-2-32456")
exit()
"""
def get_cloud_file_list(config, storage_client, client):
    cloud = []
    clients = []
    raw = []
    annotation = []
    # Find client names
    blobs = storage_client.list_blobs(config["google_storage_bucket"])
    for blob in blobs:
        if len(blob.name.split("/")) == 3 and blob.name.split("/")[2] == "":
            clients.append(blob.name.split("/")[1])
            raw.append([])
            annotation.append([])
    blobs = storage_client.list_blobs(config["google_storage_bucket"])
    for blob in blobs:
        split = blob.name.split("/")
        if (
            len(split) == 4
            and split[1] in clients
            and split[3] != ""
            and split[3] != "inspected.json"
            and split[3] != "xref.json"
        ):
            if split[2] == "raw":
                raw[clients.index(split[1])].append(split[3])
            if split[2] == "annotation":
                annotation[clients.index(split[1])].append(split[3])
    for client in clients:
        a = {
            "name": client,
            "raw": raw[clients.index(client)],
            "annotation": annotation[clients.index(client)],
        }
        cloud.append(a)
    return cloud

def get_destination_file_list(config):
    base = config["destination_directory"]
    clients = [f for f in listdir(base) if not isfile(join(base, f))]
    destination = []
    raw = []
    annotation = []
    face = []
    skeleton = []
    sound = []
    for i, client in enumerate(clients):
        raw.append([])
        annotation.append([])
        face.append([])
        skeleton.append([])
        sound.append([])
        dirs = [
            f
            for f in listdir(base + client + "/")
            if not isfile(join(base + client + "/", f))
        ]
        for dir in dirs:
            if dir == "raw":
                files = [
                    f
                    for f in listdir(base + client + "/raw/")
                    if isfile(join(base + client + "/raw/", f))
                ]
                for file in files:
                    if "inspected" not in file:
                        raw[i].append(file)
            if dir == "annotation":
                files = [
                    f
                    for f in listdir(base + client + "/annotation/")
                    if isfile(join(base + client + "/annotation/", f))
                ]
                for file in files:
                    if "xref" not in file:
                        annotation[i].append(file)
            if dir == "face":
                files = [f for f in listdir(base + client + "/face/")]
                for file in files:
                    face[i].append(file)
            if dir == "skeleton":
                files = [f for f in listdir(base + client + "/skeleton/")]
                for file in files:
                    skeleton[i].append(file)
            if dir == "sound":
                files = [f for f in listdir(base + client + "/sound/")]
                for file in files:
                    sound[i].append(file)
        a = {
            "name": client,
            "raw": raw[i],
            "annotation": annotation[i],
            "face": face[i],
            "skeleton": skeleton[i],
            "sound": sound[i]
        }
        destination.append(a)
    return destination


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config file", default="config.json")
args = parser.parse_args()

config = json.load(open(args.config))
cuda_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Building training data -------------------")
print("Google storage: ", config["google_storage_bucket"])
print("Dest. directory: ", config["destination_directory"])
print("Update interval (0 - once): ", config["update_interval_hours"])
print("Postprocess all: ", config["postprocess_all"] == 1)
print("Postprocess: ", config["postprocess"])
print("Only postprocess: ", config["postprocess_only"] == 1)
print("Display face: ", config["display_face"] == 1)
print("Clients to process: ", config["client"])
print("------------------------------------------")
while True:
    if not config["postprocess_only"]:
        storage_client = storage.Client.from_service_account_json(
            config["google_storage_key"]
        )
        bucket = storage_client.bucket(config["google_storage_bucket"])

        cloud_clients = get_cloud_file_list(config, storage_client)
        dest_clients = get_destination_file_list(config)
        file_changes = False
        changes = []
        d_idx = 0
        for c_idx, client in enumerate(cloud_clients):
            if client["name"] == config["client"] or config["client"] == "all":
                added_files = []
                deleted_files = []
                raw_files = []
                found = [f for f in dest_clients if client["name"] == f["name"]]
                if not found:
                    # Create the directory structure for new client
                    print("Creating directory structure for ", client["name"])
                    os.mkdir(config["destination_directory"] + client["name"])
                    os.mkdir(config["destination_directory"] + client["name"] + "/raw")
                    os.mkdir(
                        config["destination_directory"] + client["name"] + "/annotation"
                    )
                    a = {"name": client["name"], "raw": [], "annotation": []}
                    dest_clients.append(a)
                d_idx = [
                    i for i, f in enumerate(dest_clients) if client["name"] == f["name"]
                ][0]

                # Check if there are files to be deleted from the annotation directory
                if len(dest_clients[d_idx]["annotation"]) > 0:
                    for ann in dest_clients[d_idx]["annotation"]:
                        if ann not in client["annotation"]:
                            print("Removing: ", ann, " from client: ", client["name"])
                            os.remove(
                                config["destination_directory"]
                                + client["name"]
                                + "/annotation/"
                                + ann
                            )
                            deleted_files.append("/annotation/" + ann)
                            file_changes = True
                # Check if there are files to be added to the annotation directory
                if len(cloud_clients[c_idx]["annotation"]):
                    for i, ann in enumerate(cloud_clients[c_idx]["annotation"]):
                        if ann not in dest_clients[d_idx]["annotation"]:
                            print("Adding: ", ann, " to client: ", client["name"])
                            cloud_file = (
                                "annotation/" + client["name"] + "/annotation/" + ann
                            )
                            dest_file = (
                                config["destination_directory"]
                                + client["name"]
                                + "/annotation/"
                                + ann
                            )
                            context_blob = bucket.blob(cloud_file)
                            context_blob.download_to_filename(dest_file)
                            added_files.append("/annotation/" + ann)
                            file_changes = True
                # Create a list of active raw files referenced in the annotation file
                ann_files = [
                    f
                    for f in cloud_clients[c_idx]["annotation"]
                    if ".json" in f and "xref" not in f
                ]
                for ann_file in ann_files:
                    json_file = (
                        config["destination_directory"]
                        + client["name"]
                        + "/annotation/"
                        + ann_file
                    )
                    with open(json_file, mode="r+") as jsonFile:
                        annotation = json.load(jsonFile)
                        base_file = annotation["video"].split("/")[-1].split(".")[0]
                        raw_files.append(base_file + ".json")
                        raw_files.append(base_file + ".mp4")
                        raw_files.append(base_file + ".wav")
                        raw_files.append(base_file + "_post.wav")
                        raw_files.append(base_file + "_depth.mp4")
                        raw_files.append(base_file + ".npy.gz")

                # Check if there are files to be deleted from the raw directory
                if len(dest_clients[d_idx]["raw"]):
                    for ann in dest_clients[d_idx]["raw"]:
                        if ann not in client["raw"] or ann not in raw_files:
                            print("Removing: ", ann, " from client: ", client["name"])
                            os.remove(
                                config["destination_directory"]
                                + client["name"]
                                + "/raw/"
                                + ann
                            )
                            deleted_files.append("/raw/" + ann)
                            file_changes = True

                # Check if there are files to be added to the raw directory
                if len(cloud_clients[c_idx]["raw"]):
                    for i, ann in enumerate(cloud_clients[c_idx]["raw"]):
                        if ann not in dest_clients[d_idx]["raw"] and ann in raw_files:
                            print("Adding: ", ann, " to client: ", client["name"])
                            cloud_file = "annotation/" + client["name"] + "/raw/" + ann
                            dest_file = (
                                config["destination_directory"]
                                + client["name"]
                                + "/raw/"
                                + ann
                            )
                            context_blob = bucket.blob(cloud_file)
                            context_blob.download_to_filename(dest_file)
                            added_files.append("/raw/" + ann)
                            file_changes = True

                a = {"name": client["name"], "deleted": deleted_files, "added": added_files}
                changes.append(a)

        # Only save changes.json if there has been changes
        if file_changes:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if isfile(config["destination_directory"] + "labels.json"):
                os.rename(
                    config["destination_directory"] + "labels.json",
                    config["destination_directory"]
                    + "labels-"
                    + timestamp
                    + ".json",
                )
            if isfile(config["destination_directory"] + "context.json"):
                os.rename(
                    config["destination_directory"] + "context.json",
                    config["destination_directory"]
                    + "context-"
                    + timestamp
                    + ".json",
                )
            cloud_file = "annotation/labels.json"
            dest_file = config["destination_directory"] + "labels.json"
            context_blob = bucket.blob(cloud_file)
            context_blob.download_to_filename(dest_file)
            cloud_file = "annotation/context.json"
            dest_file = config["destination_directory"] + "context.json"
            context_blob = bucket.blob(cloud_file)
            context_blob.download_to_filename(dest_file)

            with open(
                config["destination_directory"] + "changes-" + timestamp + ".json",
                "w",
            ) as outfile:
                json.dump(changes, outfile, indent=4)

    # Check if face and skeleton files are to be generated
    feature = FeatureDetector(config["models_directory"], cuda_device, face_detection=config["face_detection_method"])
    # Loop through all clients raw files and postprocess data according to settings
    dest_clients = get_destination_file_list(config)
    if config["postprocess_all"]:
        # Delete all img and json directories
        for i, client in enumerate(dest_clients):
            if config["client"] == "all" or config["client"] == client["name"]:
                for directory_name in ["face", "skeleton", "sound"]:
                    directory = os.path.join(config["destination_directory"], client["name"], directory_name)
                    if os.path.exists(directory) and directory_name in config["postprocess"]:
                        print("Deleting ", directory)
                        os.system("rm -r " + directory)
                time.sleep(1)
    dest_clients = get_destination_file_list(config)
    for i, client in enumerate(dest_clients):
        if config["client"] == "all" or config["client"] == client["name"]:
            print("Postprocessing video for: ", client["name"])
            # Check if face, skeleton and sound directories exist, create if not
            dirs = listdir(config["destination_directory"] + client["name"] + "/")
            for directory_name in ["face", "skeleton", "sound"]:
                if not directory_name in dirs:
                    directory = os.path.join(config["destination_directory"], client["name"], directory_name)
                    print("Creating dir: ", directory)
                    os.mkdir(directory)
            time.sleep(1)
            # Loop through all raw files
            raw_files = [f for f in client["raw"] if "_post.wav" in f]
            for raw_file in raw_files:
                file_id = raw_file.split("_")[0]
                if "face" in config["postprocess"] or "skeleton" in config["postprocess"]:
                    # fetch intrinsics
                    filename = os.path.join(config["destination_directory"], client["name"], "raw", file_id+".json")
                    with open(filename, "r") as fp:
                        raw_info = json.load(fp)
                    if "intr_ppx" in raw_info:
                        intrinsics = (raw_info["intr_ppx"], raw_info["intr_ppy"], raw_info["intr_fx"], raw_info["intr_fy"])
                    else:
                        # Use default setting, intrinsics from a Intel realsense D435. All new recordings shall have intrinsics
                        intrinsics = (624.8223266, 361.023498, 926.580810, 925.832397)

                    # Generate face and skeleton frames
                    if not file_id in client["face"]:
                        if "target_person" in raw_info and len(raw_info["target_person"]) > 0:
                            # Initialize tracking
                            print("Raw info: ", raw_info)
                            target_bbox = raw_info["target_person"]
                            feature.init_tracker(np.array([target_bbox]))
                            print("Tracking initialized with bbox: ", target_bbox)

                        face_dir = os.path.join(config["destination_directory"], client["name"], "face", file_id)
                        skeleton_dir = os.path.join(config["destination_directory"], client["name"], "skeleton", file_id)
                        if not os.path.exists(face_dir):
                            os.mkdir(face_dir)
                        if not os.path.exists(skeleton_dir):
                            os.mkdir(skeleton_dir)
                        # Get the face image and store them frame by frame
                        video_file = os.path.join(config["destination_directory"], client["name"], "raw", file_id + ".mp4")
                        video_frames = load_video_frames(video_file)
                        # Handle both color compression and depth data
                        depth_file = os.path.join(config["destination_directory"], client["name"], "raw", file_id + "_depth.mp4")
                        if os.path.isfile(depth_file):
                            depth_frames = load_video_frames(depth_file)
                            raw_depth = False
                        else:
                            depth_frames = []
                            depth_file = os.path.join(config["destination_directory"], client["name"], "raw",
                                                      file_id + ".npy")
                            if not os.path.isfile(depth_file):
                                depth_file_zip = os.path.join(config["destination_directory"], client["name"], "raw",
                                                       file_id + ".npy.gz")
                                if os.path.isfile(depth_file_zip):
                                    print("Start extracting: ", depth_file_zip)
                                    os.system("gunzip " + depth_file_zip)
                            if os.path.isfile(depth_file):
                                depth_raw_frames = np.load(depth_file, allow_pickle=True)
                                # Raw depth is saved with shape: (no_of_frames * frame_height, frame_width)
                                frame_height = video_frames[0].shape[0]
                                for i in range(len(video_frames)):
                                    depth_frames.append(depth_raw_frames[i * frame_height:(i+1)*frame_height, :])
                                raw_depth = True
                            else:
                                # If no depth data exsist
                                print("No depth data is found for: ", video_file)
                        print(f"Generating frames for: {video_file} frames = {len(video_frames)}")
                        no_of_frames = min(len(video_frames), len(depth_frames))
                        for frame_no in range(no_of_frames):
                            if raw_depth:
                                depth = depth_frames[frame_no]
                            else:
                                if "min_depth" in raw_info:
                                    depth = decode_colorized(depth_frames[frame_no], raw_info["min_depth"], raw_info["max_depth"], use_disparity=True)
                                else:
                                    # 1.0 to 5.0 was the values used as default before
                                    depth = decode_colorized(depth_frames[frame_no], 1.0, 5.0, use_disparity=True)

                            person_bbox, face_bbox, face_marks, pose2d, pose3d = feature.detect_features(video_frames[frame_no],
                                                                                                           depth,
                                                                                                           intrinsics)
                            #                        frame_fts   = feature.detect_features(video_frames[frame_no], depth_frames[frame_no])
                            print("Frame ", frame_no, "Body box: ", person_bbox is not None,
                                  "Face box: ", face_bbox is not None)

                            if face_bbox is not None:
                                face = video_frames[frame_no][face_bbox[1]:face_bbox[3], \
                                       face_bbox[0]:face_bbox[2],:]
                                img = face
                                valid_face = True
                            else:
                                face = None
                                img = np.zeros((100, 100, 3))
                                valid_face = False

                            if config["display_face"] and (img.shape[0]*img.shape[1]) > 0:
                                cv2.imshow("image", img)
                                cv2.waitKey(1)

                            if "face" in config["postprocess"]:
                                with open(face_dir + "/" + str(frame_no) + ".data", "wb") as f:
                                    pickle.dump(FaceData(valid_face, face, face_marks), f)

                            if pose3d is not None:
                                valid_skeleton = True
                            else:
                                valid_skeleton = False
                            if "skeleton" in config["postprocess"]:
                                with open(skeleton_dir + "/" + str(frame_no) + ".data", "wb") as f:
                                    pickle.dump(SkeletonData(valid_skeleton, pose3d), f)
                if "sound" in config["postprocess"]:
                    if not file_id in client["sound"]:
                        sound_dir = os.path.join(config["destination_directory"], client["name"], "sound", file_id)
                        os.mkdir(sound_dir)
                        # Save the sound features as numpy arrays
                        # Fetch configuration varibles
                        sc = json.load(open(config["sensor_configuration_file"]))["Sound"]
                        sound_file = config["destination_directory"] + client["name"] + "/raw/" + file_id + ".wav"
                        data, sr = sf.read(sound_file)
                        sample_buffer = np.zeros(int(sc["feature_size"]*sc["sample_rate"]))
                        feature_no = 0
                        frame_no = 0
                        fps = json.load(open(config["sensor_configuration_file"]))["Camera"]["fps"]
                        frame_time = 1/fps
                        fts = np.zeros((13), dtype=float)
                        for idx in range(0, data.shape[0] - int(sc["feature_size"] * sc["sample_rate"]), int(sc["feature_overlap"] * sc["sample_rate"])):
                            sample_buffer = data[idx:idx+int(sc["feature_size"] * sc["sample_rate"]), 0]
                            fts = sound_feature(sample_buffer, sc["sample_rate"], sc["window_size"],
                                                    sc["step_size"], sc["feature_size"], sc["n_mfcc"])
                            # Save feature values for each frame
                            while frame_no * frame_time < feature_no * sc["feature_overlap"]:
                                np.save(sound_dir + "/" + str(frame_no) + ".npy", fts)
                                frame_no += 1
                            feature_no += 1
                        zero = np.zeros((fts.shape), dtype=float)
                        np.save(sound_dir + "/zero.npy", zero)
                # If the skeleton or face directory is empty, remove the directories


    if config["update_interval_hours"] != 0:
        time.sleep(3600 * config["update_interval_hours"])
    else:
        exit(1)
    """
