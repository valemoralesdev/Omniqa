import pydicom

def extract_dicom_metadata(filepath):
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    return ds.to_json_dict()
