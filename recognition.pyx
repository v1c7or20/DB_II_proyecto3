import os
import face_recognition
from rtree import index
import numpy as np
import json
import time


def get_files(directory):
    path = directory

    files = []

    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    return files


def calculate_vectors(files_to_compute):
    vectors_pictures = []

    for file in files_to_compute:
        name_picture = file.split('/')[3]
        picture_recognized = face_recognition.load_image_file(file)
        picture_encoding = face_recognition.face_encodings(picture_recognized)
        if len(picture_encoding) > 0:
            picture_encoding = picture_encoding[0]
            vectors_pictures.append((name_picture, picture_encoding))
    return vectors_pictures


def create_index(vector, name):
    p = index.Property()
    p.dimension = 128  # D
    p.buffering_capacity = 3  # M
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    idx = index.Index(name, properties=p)
    dato_ind = {}
    id_data = 0
    for data in vector:
        dato_ind[id_data] = data[0]
        idx.insert(id_data, np.append(data[1], data[1]))
        id_data = id_data + 1
    jsonified = json.dumps(dato_ind)
    file1 = open("ind_data" + name + '.txt', "w")
    file1.write(jsonified)
    return idx


def calculate_by_upload(uploaded):
    vectors_pictures = []
    picture_recognized = face_recognition.load_image_file(uploaded)
    picture_encoding = face_recognition.face_encodings(picture_recognized)
    if len(picture_encoding) > 0:
        picture_encoding = picture_encoding[0]
    return uploaded, picture_encoding

def knn_seq(vector_data, to_look, k):
    result = []
    for index, row in vector_data:
        d = sum((row - to_look)**2)**0.5
        result.append((index, d))
    result.sort(key=lambda elem: elem[1])
    return result[:k]

def range_search(vector_data, to_compare, r):
    result = []
    for index, row in vector_data:
        d = sum((row - to_compare)**2)**0.5
        if d <= r:
            result.append((index,d))
    result.sort(key=lambda elem: elem[1])
    return result

if __name__ == "__main__":
    files = get_files('./lfw_100')
    vectors = calculate_vectors(files)
    ind = create_index(vectors, 'ind_100')
    start = time.time()
    for j in range(10):
        for i in range(10):
            res = list(ind.nearest(coordinates=np.append(vectors[i*10][1],vectors[i*10][1]), num_results=8))
            print(res)
    end = time.time()
    print("Time consumed in working: ",end - start)
    start = time.time()
    for j in range(10):
        for i in range(10):
            res = knn_seq(vectors, vectors[i*10][1], 8)
            print(res)
    end = time.time()
    print("Time consumed in working: ",end - start)