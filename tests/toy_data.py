import pandas as pd

from ml_for_malaria.schemas import CleanedTrainingData


def toy_binary_df() -> pd.DataFrame:
    inactive = [
        "CCO",
        "CCN",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CCOCC",
        "CC(C)O",
        "CC(C)C",
        "CCCCO",
        "C1CCCCC1",
        "CCS",
        "CCBr",
        "CCOC",
        "CCCCCCC",
    ]
    active = [
        "c1ccccc1",
        "c1ccc(C)cc1",
        "c1ccc(O)cc1",
        "c1ccc(N)cc1",
        "c1ccc(F)cc1",
        "c1ccc(Cl)cc1",
        "c1ccncc1",
        "c1cccnc1",
        "CC(=O)O",
        "NCCO",
        "O=C(O)c1ccccc1",
        "COc1ccccc1",
        "CCNc1ccccc1",
        "c1ccc(Br)cc1",
        "c1ccc(CC)cc1",
    ]
    return pd.DataFrame(
        {
            CleanedTrainingData.SMILES: inactive + active,
            CleanedTrainingData.LABEL: [0] * len(inactive) + [1] * len(active),
        }
    )
