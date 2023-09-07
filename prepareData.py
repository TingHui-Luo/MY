import os
import csv
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import scipy.sparse as sp
import dgl


infNum = 1e-6

# origin data
data = np.load('./data/pemsd7-m/adj.npz')
dataArray = csc_matrix((data['data'], data['indices'], data['indptr']), shape=(228, 228)).toarray()
# print(data.files)
# print(len(data['indptr'])-1)
# print(data['format'])
# print(data['shape'])
# print(dataArray.min() + dataArray.max())
newN = 10


# 将矩阵转化为CSC格式的三个数组
def matrix_to_csc(matrix):
    if matrix.shape[0] == 0:
        return [], [], []

    rows, cols = matrix.shape
    non_zero_indices = np.transpose(np.nonzero(matrix))  # 获取非零元素的坐标并转置
    non_zero_indices_sorted = non_zero_indices[np.argsort(non_zero_indices[:, 1])]
    # print(non_zero_indices_sorted)
    # 构建CSC格式的三个数组
    indptr = [0]
    indices = []
    data = []

    current_col = 0
    for row_idx, col_idx in non_zero_indices_sorted:

        # 更新indptr数组
        while col_idx > current_col:
            indptr.append(len(indices))
            current_col += 1

        # 存储非零元素的行索引和值
        indices.append(row_idx)
        data.append(matrix[row_idx, col_idx])

    indptr.append(len(indices))
    current_col += 1
    return indptr, indices, data


# 保存前newN个传感器的数据
def createData(newN):

    # adj.npz
    newData = dataArray[:newN, :newN]
    indptr, indices, data = matrix_to_csc(newData)

    # newDataArray = csc_matrix((data, indices, indptr), shape=(newN, newN)).toarray()
    if os.path.exists(f"./data/pemsd7-m-{newN}"):
        pass
    else:
        os.mkdir(f"./data/pemsd7-m-{newN}")

    np.savez(f"./data/pemsd7-m-{newN}/adj.npz", indices=indices, indptr=indptr, format="csc", shape=[newN, newN],
             data=data)
    print(f"finish adj.npz : {newN} nodes")

    # vel.csv
    newDataVel = []
    with open('./data/pemsd7-m/vel.csv', 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            newDataVel.append(row[:newN])
    # print(newDataVel)

    with open(f"./data/pemsd7-m-{newN}/vel.csv", 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(newDataVel)

    print(f"finish vel.csv : {newN} nodes")


def tsData(newN):
    newData = np.load(f"./data/pemsd7-m-{newN}/adj.npz")
    newDataArray = csc_matrix((newData['data'], newData['indices'], newData['indptr']), shape=(newN, newN)).toarray()
    print(newDataArray)


def min_max_scaling(data):
    data_min = min(data)
    data_max = max(data)
    normalized_data = [(x - data_min) / (data_max - data_min) for x in data]
    return normalized_data


def createGrid(gridN):
    newDataInfo = []
    gridN = float(gridN)

    with open('./data/PeMSD7_M_Station_Info.csv', 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            newDataInfo.append((row[0], row[-2], row[-1]))
        # delelt csv head
        newDataInfo.pop(0)

    newDataInfo = newDataInfo[:newN]

    # Step 1: Find the bounding box of the points
    x_min = min(float(node[1]) for node in newDataInfo)
    x_max = max(float(node[1]) for node in newDataInfo)
    y_min = min(float(node[2]) for node in newDataInfo)
    y_max = max(float(node[2]) for node in newDataInfo)

    temData = []

    for node in newDataInfo:
        temX = (float(node[1]) - x_min) / (x_max - x_min)

        temY = (float(node[2]) - y_min) / (y_max - y_min)
        temData.append((node[0], temX, temY))

    newDataInfo = temData

    # Step 2: Calculate the grid side length
    x_range = 1.0
    y_range = 1.0
    grid_side_length_x = x_range / gridN
    grid_side_length_y = y_range / gridN

    # Step 3: Partition points into grids
    grids = [[] for _ in range(int(gridN) * int(gridN))]

    for node in newDataInfo:
        point_x = float(node[1])
        point_y = float(node[2])
        # print(point_x)
        # print(point_y)
        # print(grid_side_length_x + infNum)
        # print(point_x - x_min)
        col_index = math.floor(point_x / (grid_side_length_x + infNum))
        row_index = math.floor(point_y / (grid_side_length_y + infNum))
        grid_index = col_index + int(gridN) * row_index
        # print(grid_index)
        grids[grid_index].append(node[0])


    if os.path.exists(f"./data/pemsd7-m-{newN}/grid"):
        pass
    else:
        os.mkdir(f"./data/pemsd7-m-{newN}/grid")


    for i, grid in enumerate(grids):
        with open(f"./data/pemsd7-m-{newN}/grid/grid{i}.csv", 'w', newline='', encoding='utf-8') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows([[num] for num in grid])

    for i in range(int(gridN) * int(gridN)):
        with open(f"./data/pemsd7-m-{newN}/grid/grid{i}.csv", 'r', newline='', encoding='utf-8') as f:
            csvreader = csv.reader(f)
            dataMask = []
            for row in csvreader:
                dataMask.append(int(row[0]))
            createGridData(i, dataMask)
    return grids


def createGridData(gridIndex, dataMask):
    newData = np.load(f"./data/pemsd7-m-{newN}/adj.npz")
    newDataArray = csc_matrix((newData['data'], newData['indices'], newData['indptr']), shape=(newN, newN)).toarray()
    temData = []
    for i in dataMask:
        temDataRow = []
        for j in dataMask:
            temDataRow.append(newDataArray[i][j])
        temData.append(temDataRow)

    temData = np.array(temData)

    # adj.npz
    indptr, indices, data = matrix_to_csc(temData)

    if os.path.exists(f"./data/pemsd7-m-{newN}"):
        pass
    else:
        os.mkdir(f"./data/pemsd7-m-{newN}")


    if os.path.exists(f"./data/pemsd7-m-{newN}/adj"):
        pass
    else:
        os.mkdir(f"./data/pemsd7-m-{newN}/adj")


    np.savez(f"./data/pemsd7-m-{newN}/adj/adj{gridIndex}.npz", indices=indices, indptr=indptr, format="csc",
             shape=[len(dataMask), len(dataMask)],
             data=data)
    print(f"finish adj.npz : {gridIndex} grids")



    # vel.csv
    newDataVel = []
    with open(f'./data/pemsd7-m-{newN}/vel.csv', 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            newDataVel.append(row)
    newDataVel = np.array(newDataVel)

    if os.path.exists(f"./data/pemsd7-m-{newN}/vel"):
        pass
    else:
        os.mkdir(f"./data/pemsd7-m-{newN}/vel")

    temData = []
    for row in newDataVel:
        temDataRow = []
        for j in dataMask:
            temDataRow.append(row[j])
        temData.append(temDataRow)

    with open(f"./data/pemsd7-m-{newN}/vel/vel{gridIndex}.csv", 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(temData)

    print(f"finish vel.csv : {gridIndex} nodes")


def processDistDGL(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    data = np.load(os.path.join(dataset_path, 'adj.npz'))
    n_vertex = int(dataset_name.split('-')[-1])
    adj_matrix = csc_matrix((data['data'], data['indices'], data['indptr']), shape=(n_vertex, n_vertex)).toarray()

    sp_mx = sp.coo_matrix(adj_matrix)

    g = dgl.from_scipy(sp_mx)
    print(g)
    print(g.ndata)
    dgl.distributed.partition_graph(g, 'test', 2, '/tmp/test')
if __name__ == "__main__":


    # 创建小批量的数据
    # createData(newN)
    # tsData(newN)


    # 根据经纬度划分网格数据 （已遗弃）
    # grids = createGrid(2)


    # 处理DistDGL的数据
    processDistDGL("pemsd7-m-10")
    pass