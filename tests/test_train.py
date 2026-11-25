import pandas as pd

from src.train import DERNIERE_ANNEE_TRAIN, decouper


def test_decouper_separe_par_annee():
    df = pd.DataFrame({"Year": [2007, 2008, 2009, 2010, 2011, 2013]})

    train, test = decouper(df)

    assert (train["Year"] <= DERNIERE_ANNEE_TRAIN).all()
    assert (test["Year"] > DERNIERE_ANNEE_TRAIN).all()


def test_decouper_ne_perd_aucune_ligne():
    df = pd.DataFrame({"Year": [2007, 2008, 2009, 2010, 2011, 2013]})

    train, test = decouper(df)

    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(test.index)