import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
HCPex_DIR = r"D:\临时使用文件夹\NSD_Project\data\HCPex\HCPex_v1.0"
ATLAS_2MM = os.path.join(HCPex_DIR, "HCPex_2mm.nii")
LOOKUP_TABLE = os.path.join(HCPex_DIR, "HCPex_LookUpTable.txt")
OUTPUT_ATLAS = os.path.join(HCPex_DIR, "HCPex_22Systems.nii")
OUTPUT_LABELS = os.path.join(HCPex_DIR, "HCPex_22Systems_Labels.csv")
OUTPUT_MAPPING = os.path.join(HCPex_DIR, "HCPex_360_to_22_Mapping.csv")

# ==================== 1. 根据论文 Table 1 创建 Cortex ID 映射 ====================
print("=" * 60)
print("1. 创建 360 皮层区域 → 22 皮层系统 映射")
print("=" * 60)

# 根据论文 Table 1 的 Reordered ID 和 Cortex ID 创建完整映射
# 左半球: 1-180, 右半球: 181-360
# Cortex ID 1-22 对应 22 个皮层系统
cortex_id_mapping = {}

# Cortex ID 1: Primary_Visual (Reordered ID 1, 181)
for id_ in [1, 181]:
    cortex_id_mapping[id_] = 1

# Cortex ID 2: Early_Visual (Reordered ID 2-4, 182-184)
for id_ in list(range(2, 5)) + list(range(182, 185)):
    cortex_id_mapping[id_] = 2

# Cortex ID 3: Dorsal_Stream_Visual (Reordered ID 5-10, 185-190)
for id_ in list(range(5, 11)) + list(range(185, 191)):
    cortex_id_mapping[id_] = 3

# Cortex ID 4: Ventral_Stream_Visual (Reordered ID 11-17, 191-197)
for id_ in list(range(11, 18)) + list(range(191, 198)):
    cortex_id_mapping[id_] = 4

# Cortex ID 5: MT+_Complex (Reordered ID 18-26, 198-206)
for id_ in list(range(18, 27)) + list(range(198, 207)):
    cortex_id_mapping[id_] = 5

# Cortex ID 6: SomaSens_Motor (Reordered ID 27-31, 207-211)
for id_ in list(range(27, 32)) + list(range(207, 212)):
    cortex_id_mapping[id_] = 6

# Cortex ID 7: ParaCentral_MidCing (Reordered ID 32-40, 212-220)
for id_ in list(range(32, 41)) + list(range(212, 221)):
    cortex_id_mapping[id_] = 7

# Cortex ID 8: Premotor (Reordered ID 41-47, 221-227)
for id_ in list(range(41, 48)) + list(range(221, 228)):
    cortex_id_mapping[id_] = 8

# Cortex ID 9: Posterior_Opercular (Reordered ID 48-52, 228-232)
for id_ in list(range(48, 53)) + list(range(228, 233)):
    cortex_id_mapping[id_] = 9

# Cortex ID 10: Early_Auditory (Reordered ID 53-59, 233-239)
for id_ in list(range(53, 60)) + list(range(233, 240)):
    cortex_id_mapping[id_] = 10

# Cortex ID 11: Auditory_Association (Reordered ID 60-67, 240-247)
for id_ in list(range(60, 68)) + list(range(240, 248)):
    cortex_id_mapping[id_] = 11

# Cortex ID 12: Insula_FrontalOperc (Reordered ID 68-79, 248-259)
for id_ in list(range(68, 80)) + list(range(248, 260)):
    cortex_id_mapping[id_] = 12

# Cortex ID 13: Medial_Temporal (Reordered ID 80-87, 260-267)
for id_ in list(range(80, 88)) + list(range(260, 268)):
    cortex_id_mapping[id_] = 13

# Cortex ID 14: Lateral_Temporal (Reordered ID 88-95, 268-275)
for id_ in list(range(88, 96)) + list(range(268, 276)):
    cortex_id_mapping[id_] = 14

# Cortex ID 15: TPO (Reordered ID 96-100, 276-280)
for id_ in list(range(96, 101)) + list(range(276, 281)):
    cortex_id_mapping[id_] = 15

# Cortex ID 16: Superior_Parietal (Reordered ID 101-110, 281-290)
for id_ in list(range(101, 111)) + list(range(281, 291)):
    cortex_id_mapping[id_] = 16

# Cortex ID 17: Inferior_Parietal (Reordered ID 111-120, 291-300)
for id_ in list(range(111, 121)) + list(range(291, 301)):
    cortex_id_mapping[id_] = 17

# Cortex ID 18: Posterior_Cingulate (Reordered ID 121-133, 301-313)
for id_ in list(range(121, 134)) + list(range(301, 314)):
    cortex_id_mapping[id_] = 18

# Cortex ID 19: AntCing_MedPFC (Reordered ID 134-149, 314-329)
for id_ in list(range(134, 150)) + list(range(314, 330)):
    cortex_id_mapping[id_] = 19

# Cortex ID 20: OrbPolaFrontal (Reordered ID 150-158, 330-338)
for id_ in list(range(150, 159)) + list(range(330, 339)):
    cortex_id_mapping[id_] = 20

# Cortex ID 21: Inferior_Frontal (Reordered ID 159-167, 339-347)
for id_ in list(range(159, 168)) + list(range(339, 348)):
    cortex_id_mapping[id_] = 21

# Cortex ID 22: Dorsolateral_Prefrontal (Reordered ID 168-180, 348-360)
for id_ in list(range(168, 181)) + list(range(348, 361)):
    cortex_id_mapping[id_] = 22

print(f"映射关系创建完成，共 {len(cortex_id_mapping)} 个皮层区域")
print(f"Cortex ID 范围：{min(cortex_id_mapping.values())} - {max(cortex_id_mapping.values())}")

# ==================== 2. 22 个皮层系统名称 ====================
cortex_system_names = {
    1: "Primary_Visual",
    2: "Early_Visual",
    3: "Dorsal_Stream_Visual",
    4: "Ventral_Stream_Visual",
    5: "MT_Plus_Complex",
    6: "SomaSens_Motor",
    7: "ParaCentral_MidCing",
    8: "Premotor",
    9: "Posterior_Opercular",
    10: "Early_Auditory",
    11: "Auditory_Association",
    12: "Insula_FrontalOperc",
    13: "Medial_Temporal",
    14: "Lateral_Temporal",
    15: "TPO",
    16: "Superior_Parietal",
    17: "Inferior_Parietal",
    18: "Posterior_Cingulate",
    19: "AntCing_MedPFC",
    20: "OrbPolaFrontal",
    21: "Inferior_Frontal",
    22: "Dorsolateral_Prefrontal"
}

# ==================== 3. 360 个 ROI 名称映射 (根据论文 Table 1) ====================
roi_names = {
    # Cortex ID 1: Primary_Visual
    1: "V1_Primary_Visual_Cortex", 181: "V1_Primary_Visual_Cortex_R",
    # Cortex ID 2: Early_Visual
    2: "V2_Second_Visual_Area", 3: "V3_Third_Visual_Area", 4: "V4_Fourth_Visual_Area",
    182: "V2_Second_Visual_Area_R", 183: "V3_Third_Visual_Area_R", 184: "V4_Fourth_Visual_Area_R",
    # Cortex ID 3: Dorsal_Stream_Visual
    5: "IPS1_IntraParietal_Sulcus_Area_1", 6: "V3A_Area_V3A", 7: "V3B_Area_V3B", 8: "V6_Sixth_Visual_Area",
    9: "V6A_Area_V6A", 10: "V7_Seventh_Visual_Area",
    185: "IPS1_IntraParietal_Sulcus_Area_1_R", 186: "V3A_Area_V3A_R", 187: "V3B_Area_V3B_R",
    188: "V6_Sixth_Visual_Area_R", 189: "V6A_Area_V6A_R", 190: "V7_Seventh_Visual_Area_R",
    # Cortex ID 4: Ventral_Stream_Visual
    11: "FFC_Fusiform_Face_Complex", 12: "PIT_Posterior_InferoTemporal_complex", 13: "V8_Eighth_Visual_Area",
    14: "VMV1_VentroMedial_Visual_Area_1", 15: "VMV2_VentroMedial_Visual_Area_2",
    16: "VMV3_VentroMedial_Visual_Area_3", 17: "VVC_Ventral_Visual_Complex",
    191: "FFC_Fusiform_Face_Complex_R", 192: "PIT_Posterior_InferoTemporal_complex_R",
    193: "V8_Eighth_Visual_Area_R", 194: "VMV1_VentroMedial_Visual_Area_1_R",
    195: "VMV2_VentroMedial_Visual_Area_2_R", 196: "VMV3_VentroMedial_Visual_Area_3_R",
    197: "VVC_Ventral_Visual_Complex_R",
    # Cortex ID 5: MT+_Complex
    18: "FST_Area_FST", 19: "LO1_Area_Lateral_Occipital_1", 20: "LO2_Area_Lateral_Occipital_2",
    21: "LO3_Area_Lateral_Occipital_3", 22: "MST_Medial_Superior_Temporal_Area",
    23: "MT_Middle_Temporal_Area", 24: "PH_Area_PH", 25: "V3CD_Area_V3CD", 26: "V4t_Area_V4t",
    198: "FST_Area_FST_R", 199: "LO1_Area_Lateral_Occipital_1_R", 200: "LO2_Area_Lateral_Occipital_2_R",
    201: "LO3_Area_Lateral_Occipital_3_R", 202: "MST_Medial_Superior_Temporal_Area_R",
    203: "MT_Middle_Temporal_Area_R", 204: "PH_Area_PH_R", 205: "V3CD_Area_V3CD_R", 206: "V4t_Area_V4t_R",
    # Cortex ID 6: SomaSens_Motor
    27: "1_Area_1", 28: "2_Area_2", 29: "3a_Area_3a", 30: "3b_Primary_Sensory_Cortex", 31: "4_Primary_Motor_Cortex",
    207: "1_Area_1_R", 208: "2_Area_2_R", 209: "3a_Area_3a_R", 210: "3b_Primary_Sensory_Cortex_R",
    211: "4_Primary_Motor_Cortex_R",
    # Cortex ID 7: ParaCentral_MidCing
    32: "23c_Area_23c", 33: "24dd_Dorsal_Area_24d", 34: "24dv_Ventral_Area_24d", 35: "5L_Area_5L",
    36: "5m_Area_5m", 37: "5mv_Area_5m_ventral", 38: "6ma_Area_6m_anterior", 39: "6mp_Area_6mp",
    40: "SCEF_Supplementary_and_Cingulate_Eye_Field",
    212: "23c_Area_23c_R", 213: "24dd_Dorsal_Area_24d_R", 214: "24dv_Ventral_Area_24d_R",
    215: "5L_Area_5L_R", 216: "5m_Area_5m_R", 217: "5mv_Area_5m_ventral_R", 218: "6ma_Area_6m_anterior_R",
    219: "6mp_Area_6mp_R", 220: "SCEF_Supplementary_and_Cingulate_Eye_Field_R",
    # Cortex ID 8: Premotor
    41: "55b_Area_55b", 42: "6a_Area_6_anterior", 43: "6d_Dorsal_area_6", 44: "6r_Rostral_Area_6",
    45: "6v_Ventral_Area_6", 46: "FEF_Frontal_Eye_Fields", 47: "PEF_Premotor_Eye_Field",
    221: "55b_Area_55b_R", 222: "6a_Area_6_anterior_R", 223: "6d_Dorsal_area_6_R", 224: "6r_Rostral_Area_6_R",
    225: "6v_Ventral_Area_6_R", 226: "FEF_Frontal_Eye_Fields_R", 227: "PEF_Premotor_Eye_Field_R",
    # Cortex ID 9: Posterior_Opercular
    48: "43_Area_43", 49: "FOP1_Frontal_Opercular_Area_1", 50: "OP1_Area_OP1_SII",
    51: "OP2_3_Area_OP2_3_VS", 52: "OP4_Area_OP4_PV",
    228: "43_Area_43_R", 229: "FOP1_Frontal_Opercular_Area_1_R", 230: "OP1_Area_OP1_SII_R",
    231: "OP2_3_Area_OP2_3_VS_R", 232: "OP4_Area_OP4_PV_R",
    # Cortex ID 10: Early_Auditory
    53: "52_Area_52", 54: "A1_Primary_Auditory_Cortex", 55: "LBelt_Lateral_Belt_Complex",
    56: "MBelt_Medial_Belt_Complex", 57: "PBelt_ParaBelt_Complex", 58: "PFcm_Area_PFcm", 59: "RI_RetroInsular_Cortex",
    233: "52_Area_52_R", 234: "A1_Primary_Auditory_Cortex_R", 235: "LBelt_Lateral_Belt_Complex_R",
    236: "MBelt_Medial_Belt_Complex_R", 237: "PBelt_ParaBelt_Complex_R", 238: "PFcm_Area_PFcm_R",
    239: "RI_RetroInsular_Cortex_R",
    # Cortex ID 11: Auditory_Association
    60: "A4_Auditory_4_Complex", 61: "A5_Auditory_5_Complex", 62: "STGa_Area_STGa",
    63: "STSda_Area_STSd_anterior", 64: "STSdp_Area_STSd_posterior", 65: "STSva_Area_STSv_anterior",
    66: "STSvp_Area_STSv_posterior", 67: "TA2_Area_TA2",
    240: "A4_Auditory_4_Complex_R", 241: "A5_Auditory_5_Complex_R", 242: "STGa_Area_STGa_R",
    243: "STSda_Area_STSd_anterior_R", 244: "STSdp_Area_STSd_posterior_R", 245: "STSva_Area_STSv_anterior_R",
    246: "STSvp_Area_STSv_posterior_R", 247: "TA2_Area_TA2_R",
    # Cortex ID 12: Insula_FrontalOperc
    68: "AAIC_Anterior_Agranular_Insula_Complex", 69: "AVI_Anterior_Ventral_Insular_Area",
    70: "FOP2_Frontal_Opercular_Area_2", 71: "FOP3_Frontal_Opercular_Area_3", 72: "FOP4_Frontal_Opercular_Area_4",
    73: "FOP5_Area_Frontal_Opercular_5", 74: "Ig_Insular_Granular_Complex", 75: "MI_Middle_Insular_Area",
    76: "PI_Para_Insular_Area", 77: "Pir_Pirform_Cortex", 78: "PoI1_Area_Posterior_Insular_1",
    79: "PoI2_Posterior_Insular_Area_2",
    248: "AAIC_Anterior_Agranular_Insula_Complex_R", 249: "AVI_Anterior_Ventral_Insular_Area_R",
    250: "FOP2_Frontal_Opercular_Area_2_R", 251: "FOP3_Frontal_Opercular_Area_3_R",
    252: "FOP4_Frontal_Opercular_Area_4_R", 253: "FOP5_Area_Frontal_Opercular_5_R",
    254: "Ig_Insular_Granular_Complex_R", 255: "MI_Middle_Insular_Area_R", 256: "PI_Para_Insular_Area_R",
    257: "Pir_Pirform_Cortex_R", 258: "PoI1_Area_Posterior_Insular_1_R", 259: "PoI2_Posterior_Insular_Area_2_R",
    # Cortex ID 13: Medial_Temporal
    80: "H_Hippocampus", 81: "PreS_PreSubiculum", 82: "EC_Entorhinal_Cortex", 83: "PeEc_Perirhinal_Ectorhinal_Cortex",
    84: "TF_Area_TF", 85: "PHA1_ParaHippocampal_Area_1", 86: "PHA2_ParaHippocampal_Area_2",
    87: "PHA3_ParaHippocampal_Area_3",
    260: "H_Hippocampus_R", 261: "PreS_PreSubiculum_R", 262: "EC_Entorhinal_Cortex_R",
    263: "PeEc_Perirhinal_Ectorhinal_Cortex_R", 264: "TF_Area_TF_R", 265: "PHA1_ParaHippocampal_Area_1_R",
    266: "PHA2_ParaHippocampal_Area_2_R", 267: "PHA3_ParaHippocampal_Area_3_R",
    # Cortex ID 14: Lateral_Temporal
    88: "PHT_Area_PHT", 89: "TE1a_Area_TE1_anterior", 90: "TE1m_Area_TE1_Middle", 91: "TE1p_Area_TE1_posterior",
    92: "TE2a_Area_TE2_anterior", 93: "TE2p_Area_TE2_posterior", 94: "TGd_Area_TG_dorsal",
    95: "TGv_Area_TG_Ventral",
    268: "PHT_Area_PHT_R", 269: "TE1a_Area_TE1_anterior_R", 270: "TE1m_Area_TE1_Middle_R",
    271: "TE1p_Area_TE1_posterior_R", 272: "TE2a_Area_TE2_anterior_R", 273: "TE2p_Area_TE2_posterior_R",
    274: "TGd_Area_TG_dorsal_R", 275: "TGv_Area_TG_Ventral_R",
    # Cortex ID 15: TPO
    96: "PSL_PeriSylvian_Language_Area", 97: "STV_Superior_Temporal_Visual_Area",
    98: "TPOJ1_Area_TemporoParietoOccipital_Junction_1", 99: "TPOJ2_Area_TemporoParietoOccipital_Junction_2",
    100: "TPOJ3_Area_TemporoParietoOccipital_Junction_3",
    276: "PSL_PeriSylvian_Language_Area_R", 277: "STV_Superior_Temporal_Visual_Area_R",
    278: "TPOJ1_Area_TemporoParietoOccipital_Junction_1_R",
    279: "TPOJ2_Area_TemporoParietoOccipital_Junction_2_R",
    280: "TPOJ3_Area_TemporoParietoOccipital_Junction_3_R",
    # Cortex ID 16: Superior_Parietal
    101: "7AL_Lateral_Area_7A", 102: "7Am_Medial_Area_7A", 103: "7PC_Area_7PC", 104: "7Pl_Lateral_Area_7P",
    105: "7Pm_Medial_Area_7P", 106: "AIP_Anterior_IntraParietal_Area", 107: "LIPd_Area_Lateral_IntraParietal_dorsal",
    108: "LIPv_Area_Lateral_IntraParietal_ventral", 109: "MIP_Medial_IntraParietal_Area",
    110: "VIP_Ventral_IntraParietal_Complex",
    281: "7AL_Lateral_Area_7A_R", 282: "7Am_Medial_Area_7A_R", 283: "7PC_Area_7PC_R",
    284: "7Pl_Lateral_Area_7P_R", 285: "7Pm_Medial_Area_7P_R", 286: "AIP_Anterior_IntraParietal_Area_R",
    287: "LIPd_Area_Lateral_IntraParietal_dorsal_R", 288: "LIPv_Area_Lateral_IntraParietal_ventral_R",
    289: "MIP_Medial_IntraParietal_Area_R", 290: "VIP_Ventral_IntraParietal_Complex_R",
    # Cortex ID 17: Inferior_Parietal
    111: "IP0_Area_IntraParietal_0", 112: "IP1_Area_IntraParietal_1", 113: "IP2_Area_IntraParietal_2",
    114: "PF_Area_PF_Complex", 115: "PFm_Area_PFm_Complex", 116: "PFop_Area_PF_Opercular",
    117: "PFt_Area_PFt", 118: "PGi_Area_PGi", 119: "PGp_Area_PGp", 120: "PGs_Area_PGs",
    291: "IP0_Area_IntraParietal_0_R", 292: "IP1_Area_IntraParietal_1_R", 293: "IP2_Area_IntraParietal_2_R",
    294: "PF_Area_PF_Complex_R", 295: "PFm_Area_PFm_Complex_R", 296: "PFop_Area_PF_Opercular_R",
    297: "PFt_Area_PFt_R", 298: "PGi_Area_PGi_R", 299: "PGp_Area_PGp_R", 300: "PGs_Area_PGs_R",
    # Cortex ID 18: Posterior_Cingulate
    121: "23d_Area_23d", 122: "31a_Area_31a", 123: "31pd_Area_31pd", 124: "31pv_Area_31p_ventral",
    125: "7m_Area_7m", 126: "d23ab_Area_dorsal_23_a_b", 127: "DVT_Dorsal_Transitional_Visual_Area",
    128: "PCV_PreCuneus_Visual_Area", 129: "POS1_Parieto_Occipital_Sulcus_Area_1",
    130: "POS2_Parieto_Occipital_Sulcus_Area_2", 131: "ProS_ProStriate_Area", 132: "RSC_RetroSplenial_Complex",
    133: "v23ab_Area_ventral_23_a_b",
    301: "23d_Area_23d_R", 302: "31a_Area_31a_R", 303: "31pd_Area_31pd_R", 304: "31pv_Area_31p_ventral_R",
    305: "7m_Area_7m_R", 306: "d23ab_Area_dorsal_23_a_b_R", 307: "DVT_Dorsal_Transitional_Visual_Area_R",
    308: "PCV_PreCuneus_Visual_Area_R", 309: "POS1_Parieto_Occipital_Sulcus_Area_1_R",
    310: "POS2_Parieto_Occipital_Sulcus_Area_2_R", 311: "ProS_ProStriate_Area_R",
    312: "RSC_RetroSplenial_Complex_R", 313: "v23ab_Area_ventral_23_a_b_R",
    # Cortex ID 19: AntCing_MedPFC
    134: "10r_Area_10r", 135: "10v_Area_10v", 136: "25_Area_25", 137: "33pr_Area_33_prime",
    138: "8BM_Area_8BM", 139: "9m_Area_9_Middle", 140: "a24_Area_a24", 141: "a24pr_Anterior_24_prime",
    142: "a32pr_Area_anterior_32_prime", 143: "d32_Area_dorsal_32", 144: "p24_Area_posterior_24",
    145: "p24pr_Area_Posterior_24_prime", 146: "p32_Area_p32", 147: "p32pr_Area_p32_prime",
    148: "pOFC_Posterior_OFC_Complex", 149: "s32_Area_s32",
    314: "10r_Area_10r_R", 315: "10v_Area_10v_R", 316: "25_Area_25_R", 317: "33pr_Area_33_prime_R",
    318: "8BM_Area_8BM_R", 319: "9m_Area_9_Middle_R", 320: "a24_Area_a24_R", 321: "a24pr_Anterior_24_prime_R",
    322: "a32pr_Area_anterior_32_prime_R", 323: "d32_Area_dorsal_32_R", 324: "p24_Area_posterior_24_R",
    325: "p24pr_Area_Posterior_24_prime_R", 326: "p32_Area_p32_R", 327: "p32pr_Area_p32_prime_R",
    328: "pOFC_Posterior_OFC_Complex_R", 329: "s32_Area_s32_R",
    # Cortex ID 20: OrbPolaFrontal
    150: "10d_Area_10d", 151: "10pp_Polar_10p", 152: "11l_Area_11l", 153: "13l_Area_13l",
    154: "47m_Area_47m", 155: "47s_Area_47s", 156: "a10p_Area_anterior_10p", 157: "OFC_Orbital_Frontal_Complex",
    158: "p10p_Area_posterior_10p",
    330: "10d_Area_10d_R", 331: "10pp_Polar_10p_R", 332: "11l_Area_11l_R", 333: "13l_Area_13l_R",
    334: "47m_Area_47m_R", 335: "47s_Area_47s_R", 336: "a10p_Area_anterior_10p_R",
    337: "OFC_Orbital_Frontal_Complex_R", 338: "p10p_Area_posterior_10p_R",
    # Cortex ID 21: Inferior_Frontal
    159: "44_Area_44", 160: "45_Area_45", 161: "47l_Area_47l", 162: "a47r_Area_anterior_47r",
    163: "IFJa_Area_IFJa", 164: "IFJp_Area_IFJp", 165: "IFSa_Area_IFSa", 166: "IFSp_Area_IFSp",
    167: "p47r_Area_posterior_47r",
    339: "44_Area_44_R", 340: "45_Area_45_R", 341: "47l_Area_47l_R", 342: "a47r_Area_anterior_47r_R",
    343: "IFJa_Area_IFJa_R", 344: "IFJp_Area_IFJp_R", 345: "IFSa_Area_IFSa_R", 346: "IFSp_Area_IFSp_R",
    347: "p47r_Area_posterior_47r_R",
    # Cortex ID 22: Dorsolateral_Prefrontal
    168: "46_Area_46", 169: "8Ad_Area_8Ad", 170: "8Av_Area_8Av", 171: "8BL_Area_8B_Lateral",
    172: "8C_Area_8C", 173: "9_46d_Area_9_46d", 174: "9a_Area_9_anterior", 175: "9p_Area_9_Posterior",
    176: "a9_46v_Area_anterior_9_46v", 177: "i6_8_Inferior_6_8_Transitional_Area",
    178: "p9_46v_Area_posterior_9_46v", 179: "s6_8_Superior_6_8_Transitional_Area",
    180: "SFL_Superior_Frontal_Language_Area",
    348: "46_Area_46_R", 349: "8Ad_Area_8Ad_R", 350: "8Av_Area_8Av_R", 351: "8BL_Area_8B_Lateral_R",
    352: "8C_Area_8C_R", 353: "9_46d_Area_9_46d_R", 354: "9a_Area_9_anterior_R", 355: "9p_Area_9_Posterior_R",
    356: "a9_46v_Area_anterior_9_46v_R", 357: "i6_8_Inferior_6_8_Transitional_Area_R",
    358: "p9_46v_Area_posterior_9_46v_R", 359: "s6_8_Superior_6_8_Transitional_Area_R",
    360: "SFL_Superior_Frontal_Language_Area_R"
}

# ==================== 4. 加载原始图谱 ====================
print("\n" + "=" * 60)
print("2. 加载原始 HCPex 图谱")
print("=" * 60)

atlas_img = nib.load(ATLAS_2MM)
atlas_data = atlas_img.get_fdata()
print(f"图谱形状：{atlas_data.shape}")
print(f"标签范围：{atlas_data.min()} - {atlas_data.max()}")
print(f"唯一标签数：{len(np.unique(atlas_data))}")

# ==================== 5. 创建 22 系统图谱 ====================
print("\n" + "=" * 60)
print("3. 创建 22 个皮层系统图谱")
print("=" * 60)

new_data = np.zeros_like(atlas_data)

# 遍历原始图谱中的每个标签值
for label_id in np.unique(atlas_data):
    if label_id == 0:
        continue

    label_id = int(label_id)

    # 皮层区域 (1-360) 映射为 Cortex ID (1-22)
    if 1 <= label_id <= 360:
        if label_id in cortex_id_mapping:
            cortex_id = cortex_id_mapping[label_id]
            new_data[atlas_data == label_id] = cortex_id
        else:
            print(f"警告：标签 {label_id} 未在映射表中")
    # 皮层下区域 (361-426) 设为 0（不包含在 22 个皮层系统中）
    elif 361 <= label_id <= 426:
        new_data[atlas_data == label_id] = 0  # 或设为 100+ 单独标记

# 创建新的 NIfTI 图像
new_atlas_img = nib.Nifti1Image(new_data, atlas_img.affine, atlas_img.header)
nib.save(new_atlas_img, OUTPUT_ATLAS)
print(f"22 系统图谱已保存：{OUTPUT_ATLAS}")

# 验证
unique_labels = np.unique(new_data)
print(f"22 系统图谱唯一标签：{unique_labels}")
print(f"有效皮层系统数：{len(unique_labels[unique_labels > 0])}")

# ==================== 6. 保存标签文件 ====================
print("\n" + "=" * 60)
print("4. 保存 22 系统标签文件")
print("=" * 60)

# 创建标签 DataFrame
labels_df = pd.DataFrame([
                             {'ID': 0, 'Name': 'Background', 'CortexID': 0},
                         ] + [
                             {'ID': cid, 'Name': name, 'CortexID': cid}
                             for cid, name in cortex_system_names.items()
                         ])

labels_df.to_csv(OUTPUT_LABELS, index=False)
print(f"标签文件已保存：{OUTPUT_LABELS}")
print("\n标签内容：")
print(labels_df.to_string(index=False))

# ==================== 7. 保存完整映射表 (360→22) ====================
print("\n" + "=" * 60)
print("5. 保存完整映射表 (360 个 ROI → 22 个系统)")
print("=" * 60)

mapping_data = []
for roi_id in range(1, 361):
    roi_name = roi_names.get(roi_id, f"Unknown_{roi_id}")
    cortex_id = cortex_id_mapping.get(roi_id, 0)
    cortex_name = cortex_system_names.get(cortex_id, "Unknown")
    hemisphere = "R" if roi_id > 180 else "L"

    mapping_data.append({
        'Reordered_ID': roi_id,
        'Hemisphere': hemisphere,
        'ROI_Name': roi_name,
        'Cortex_ID': cortex_id,
        'Cortex_System_Name': cortex_name
    })

mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv(OUTPUT_MAPPING, index=False, encoding='utf-8-sig')
print(f"完整映射表已保存：{OUTPUT_MAPPING}")

# ==================== 8. 打印映射表（方便检查）====================
print("\n" + "=" * 60)
print("📋 完整映射表 (360 个 ROI → 22 个皮层系统)")
print("=" * 60)
print(f"{'ID':<6} {'Hemi':<6} {'ROI 名称':<50} {'Cortex ID':<12} {'皮层系统名称'}")
print("-" * 120)

for row in mapping_data:
    print(f"{row['Reordered_ID']:<6} {row['Hemisphere']:<6} {row['ROI_Name']:<50} "
          f"{row['Cortex_ID']:<12} {row['Cortex_System_Name']}")

print("-" * 120)
print(f"总计：360 个皮层区域 → 22 个皮层系统")

# ==================== 9. 按皮层系统分组统计 ====================
print("\n" + "=" * 60)
print("📊 按皮层系统分组统计")
print("=" * 60)
print(f"{'Cortex ID':<12} {'皮层系统名称':<30} {'区域数':<10} {'左半球':<10} {'右半球':<10}")
print("-" * 80)

for cid in range(1, 23):
    system_name = cortex_system_names[cid]
    total_count = sum(1 for m in mapping_data if m['Cortex_ID'] == cid)
    left_count = sum(1 for m in mapping_data if m['Cortex_ID'] == cid and m['Hemisphere'] == 'L')
    right_count = sum(1 for m in mapping_data if m['Cortex_ID'] == cid and m['Hemisphere'] == 'R')
    print(f"{cid:<12} {system_name:<30} {total_count:<10} {left_count:<10} {right_count:<10}")

print("-" * 80)
print(f"总计：{len(mapping_data)} 个皮层区域")

print("\n" + "=" * 60)
print("6. 构建 Nilearn 标签字典")
print("=" * 60)

labels_dict = {0: 'Background'}
for cid, name in cortex_system_names.items():
    labels_dict[cid] = name

print(f"标签字典大小：{len(labels_dict)}")
print(f"示例：{list(labels_dict.items())[:5]}")
