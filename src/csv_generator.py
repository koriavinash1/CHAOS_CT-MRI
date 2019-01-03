import os
import numpy as np
import pandas as pd

base_path_ct = '../DATA/CT_data_batch1'
base_path_mri = '../DATA/MRI_data_batch1'

dcm = []
gt  = []

patients = os.listdir(base_path_ct)
for patient in patients:
    patient_dir = os.path.join(base_path_ct, patient)
    dcm_pth = os.path.join(patient_dir, 'DICOM_anon')
    gt_pth = os.path.join(patient_dir, 'Ground')

    slices = os.listdir(gt_pth)
    for slice_ in slices:
        slice_no = slice_.split('.')[0][-3:]
        dcm_id = 'i0'+slice_no+',0000b.dcm'

        dcm.append(os.path.join(dcm_pth, dcm_id))
        gt.append(os.path.join(gt_pth, slice_))

df = pd.DataFrame()
df['dcm_path'] = np.array(dcm)
df['gt_path'] = np.array(gt)

df = df.sample(frac=1).reset_index(drop=True)

len_ = len(df)

df[:int(0.8*len_)].to_csv('./train.csv')
df[int(0.8*len_):int(0.9*len_)].to_csv('./valid.csv')
df[int(0.9*len_):].to_csv('./test.csv')

