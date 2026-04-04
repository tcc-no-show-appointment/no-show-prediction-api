PREDICTION_LABEL_SHOW = "show"
PREDICTION_LABEL_NO_SHOW = "no-show"



# All specialty_group values produced by noshow_lib feature engineering.
# Used to pre-populate the effective_models dict with the default fallback
# for any specialty that does not yet have a dedicated trained model.
KNOWN_SPECIALTY_GROUPS = [
    "CLINICA_ESPECIALIZADA",
    "CLINICA_GERAL_E_TRIAGEM",
    "CIRURGICO_E_VASCULAR",
    "EXAMES_E_PROCEDIMENTOS",
    "MATERNO_INFANTIL",
    "ORTOPEDIA",
    "SAUDE_MENTAL",
    "TERAPIAS_REABILITACAO",
    "OUTRAS_ESPECIALIDADES",
]