from .elec import *
import os

def get_dataset_key_from_path(dataset_path):
    # 파일 이름만 추출 (디렉토리 경로 제외)
    filename = os.path.basename(dataset_path)

    # 파일 이름에 따라 데이터셋 키 반환 (파일 이름에 맞는 키를 설정)
    if 'rea_final_data' in filename:
        return 'elec'  # 'elec' 키로 처리, 필요시 다른 키로 변경 가능
    else:
        raise ValueError(f"Unknown dataset: {filename}")
    
def get_configs(dataset, name):
    elec_config_dict = {
        'ddpm_d': get_elec_ddpm_discrete_config(),
        'ddpm_c': get_elec_ddpm_cont_config(),
        'smld_d': get_elec_smld_discrete_config(),
        'smld_c': get_elec_smld_cont_config(),
        'subvpsde': get_elec_subvpsde_config()
    }

    config_dict = {
        'elec': elec_config_dict,
    }
    if dataset.endswith('.csv'):
        dataset = get_dataset_key_from_path(dataset)

    return config_dict[dataset][name]