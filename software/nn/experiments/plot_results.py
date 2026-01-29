import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
long_csv = os.path.join(ROOT, "long_train_curve.csv")
cv_csv = os.path.join(ROOT, "cv_results.csv")

out_long = os.path.join(ROOT, "long_train_curve.png")
out_cv = os.path.join(ROOT, "cv_results.png")

def plot_long():
    if not os.path.exists(long_csv):
        print(f"Missing {long_csv}")
        return 1
    df = pd.read_csv(long_csv)
    # try to infer columns
    cols = df.columns.str.lower()
    epoch_col = None
    for c in ['epoch','ep','step']:
        if c in cols:
            epoch_col = df.columns[list(cols).index(c)]
            break
    if epoch_col is None:
        epoch_col = df.columns[0]
    plt.figure(figsize=(8,4))
    if any('train_loss'==c for c in cols) or any('loss'==c for c in cols):
        loss_col = None
        for c in df.columns:
            if c.lower()=='train_loss' or c.lower()=='loss':
                loss_col = c
                break
        if loss_col:
            plt.plot(df[epoch_col], df[loss_col], label='train_loss')
    # accuracy
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    for c in acc_cols:
        plt.plot(df[epoch_col], df[c], label=c)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Long training curve')
    plt.tight_layout()
    plt.savefig(out_long)
    print('Saved', out_long)
    return 0


def plot_cv():
    if not os.path.exists(cv_csv):
        print(f"Missing {cv_csv}")
        return 1
    df = pd.read_csv(cv_csv)
    # Expect columns: candidate, fold, val_acc or similar
    cols = df.columns.str.lower()
    if 'val_acc' in cols:
        val_col = df.columns[list(cols).index('val_acc')]
    elif 'val accuracy' in cols:
        val_col = df.columns[list(cols).index('val accuracy')]
    else:
        # try any column with 'acc'
        val_candidates = [c for c in df.columns if 'acc' in c.lower()]
        val_col = val_candidates[0] if val_candidates else None
    if val_col is None:
        print('No accuracy column found in', cv_csv)
        return 1
    # candidate column
    cand_col = None
    for name in ['candidate','config','params']:
        if name in cols:
            cand_col = df.columns[list(cols).index(name)]
            break
    if cand_col is None:
        # try first column
        cand_col = df.columns[0]
    # aggregate
    agg = df.groupby(cand_col)[val_col].agg(['mean','std','count']).reset_index()
    plt.figure(figsize=(8,4))
    x = range(len(agg))
    plt.bar(x, agg['mean'], yerr=agg['std'], capsize=5)
    plt.xticks(x, agg[cand_col], rotation=45, ha='right')
    plt.ylabel(val_col)
    plt.title('CV mean val accuracy')
    plt.tight_layout()
    plt.savefig(out_cv)
    print('Saved', out_cv)
    return 0


def main():
    status1 = plot_long()
    status2 = plot_cv()
    return 0 if (status1==0 and status2==0) else 2

if __name__=='__main__':
    sys.exit(main())
