
import scipy.io

def load_data(file_path):
    mat_contents = scipy.io.loadmat(file_path)
    QML = mat_contents['QML286']
    print("QML data successfully loaded")
    return QML