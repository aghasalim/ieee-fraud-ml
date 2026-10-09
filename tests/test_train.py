from src.fraud import train


def test_final_val_auc_is_read_from_the_ablation():
    # The app shows this number; it has to be the ablation row the final
    # model's feature set came from, not a literal typed into train.py.
    assert train.final_val_auc() == 0.8839
