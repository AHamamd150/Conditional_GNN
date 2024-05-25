import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse


def list_of_strings(arg):
    return arg.split(",")


hetero_clms = [
    "p_l.first.Pt()",
    "p_l.first.Eta()",
    "p_l.first.Phi()",
    "p_l.first.E()",
    "p_b1.Pt()",
    "p_b1.Eta()",
    "p_b1.Phi()",
    "p_b1.E()",
    "p_v1.Pt()",
    "p_v1.Eta()",
    "p_v1.Phi()",
    "p_v1.E()",
    "p_l.second.Pt()",
    "p_l.second.Eta()",
    "p_l.second.Phi()",
    "p_l.second.E()",
    "p_b2.Pt()",
    "p_b2.Eta()",
    "p_b2.Phi()",
    "p_b2.E()",
    "p_v2.Pt()",
    "p_v2.Eta()",
    "p_v2.Phi()",
    "p_v2.E()",
    "pb_fromHiggs_1.Pt()",
    "pb_fromHiggs_1.Eta()",
    "pb_fromHiggs_1.Phi()",
    "pb_fromHiggs_1.E()",
    "pb_fromHiggs_2.Pt()",
    "pb_fromHiggs_2.Eta()",
    "pb_fromHiggs_2.Phi()",
    "pb_fromHiggs_2.E()",
    "p_top.M()",
    "p_tbar.M()",
    "p_top.Pt()",
    "p_top.Eta()",
    "p_top.Phi()",
    "p_top.E()",
    "p_tbar.Pt()",
    "p_tbar.Eta()",
    "p_tbar.Phi()",
    "p_tbar.E()",
    "pHiggs.Pt()",
    "pHiggs.Eta()",
    "pHiggs.Phi()",
    "pHiggs.E()",
    "p_ttbar.Pt()",
    "p_ttbar.M()",
    "p_ttbar.E()",
    "p_ttH.M()",
    "p_ll.Pt()",
    "HTb",
    "Meff",
    "MET",
    "Phimiss",
    "Cos_thetahel1",
    "Cos_thetahel2",
    "Cos_opening",
    "Cos_thetaHL",
    "Cos_thetaHLCPV",
    "Cos_wLhh",
    "Cos_wLpLh",
    "Cos_BpBh",
    "Cos_wBpBhL",
    "Cos_wBpBLpL",
    "Cos_wBpBBpB",
    "Cos_wBhh",
    "Cos_wBhLpL",
    "Cos_wBhBpB",
    "Cos_wBLpLh",
    "Cos_wLpLLpLb",
    "Cos_wLpLBpBb",
    "Cos_wLmLLmLb",
    "Cos_thetatrl1",
    "Cos_thetatrl2",
    "Cos_thetarbl1",
    "Cos_thetarbl2",
    "ulab1",
    "ulab2",
    "xlab1",
    "xlab2",
    "zlab1",
    "zlab2",
    "xrest1",
    "xrest2",
    "c1c2_kk",
    "c1c2_nn",
    "c1c2_rr",
    "c1c2_nk",
    "c1c2_kr",
    "c1c2_nr",
    "dPhill",
    "dPhi_lltt",
    "Cos_thetaS",
]


def read_clean(root_dir, alphas, frac=0.5):
    """
    Reads multiple files from given directory paths, processes and cleans the data,
    and combines it into a single DataFrame ready for analysis.

    The function reads three types of data files (HighLevel, LowLevel, Polarization) for each
    alpha value provided. It attempts to cast each alpha to float and uses it to label the data.
    If the casting fails, it assigns a random alpha value from a uniform distribution as a label.

    The resulting DataFrame is filtered by certain criteria (e.g., mass of the top particle and
    removing outliers), and then selected columns are retained for further analysis.

    Parameters:
    - root_dir (str): The directory containing the data files. Each data file is expected
      to be in the format "alpha-{alpha_value}_{DataType}.csv".
    - alphas (list of str): A list of alpha values as strings. Each value corresponds to a set
      of data files.
    - frac (float): Randomaly choose frac of events of ONLY signal. This is because there are
      many signal points and only one background point.

    Returns:
    - pandas.DataFrame: A DataFrame containing concatenated and cleaned data from all provided
      alpha values. The DataFrame includes selected features and labels for machine learning
      applications.

    Examples:
    - Example usage:
        >>> df = read_clean("/path/to/data/", ["0.1", "0.2", "0.3"])
    """
    print(
        f"Reading all files for given alpha values {alphas}. This may take a few minutes..."
    )
    parameters = [
        0.000000,
        0.261766,
        0.523599,
        0.785398,
        1.047200,
        1.309000,
        1.570800,
    ]
    result = []
    for alpha in alphas:
        try:
            a = float(alpha)
            # read the data
            high_level_data = pd.read_csv(
                root_dir + "alpha-" + alpha + "_HighLevel.csv"
            )
            low_level_data = pd.read_csv(root_dir + "alpha-" + alpha + "_LowLevel.csv")
            polarization_data = pd.read_csv(
                root_dir + "alpha-" + alpha + "_Polarization.csv"
            )
            df = pd.concat(
                [
                    low_level_data.iloc[:, 1:],
                    high_level_data.iloc[:, 1:],
                    polarization_data.iloc[:, 1:],
                ],
                axis=1,
            )

            # Clean the data
            df = df[(df["p_top.M()"] < 173 + 40) & (df["p_top.M()"] > 173 - 40)]
            # Remove outliers from all columns
            # TODO: is 4000 a good value for all observables
            df = df[df.lt(4000).all(axis=1)]
            # Select relevant columns for GNN
            df = df[hetero_clms]
            # Choose a random subset
            df = df.sample(frac=frac, random_state=42)

            df["alpha"] = np.ones(df.shape[0]) * a
            df["label"] = np.ones(df.shape[0])
            # append to the overall dataframe
            result.append(df)
        except:
            high_level_data = pd.read_csv(
                root_dir + "alpha-" + alpha + "_HighLevel.csv"
            )
            low_level_data = pd.read_csv(root_dir + "alpha-" + alpha + "_LowLevel.csv")
            polarization_data = pd.read_csv(
                root_dir + "alpha-" + alpha + "_Polarization.csv"
            )
            df = pd.concat(
                [
                    low_level_data.iloc[:, 1:],
                    high_level_data.iloc[:, 1:],
                    polarization_data.iloc[:, 1:],
                ],
                axis=1,
            )

            # Clean the data
            df = df[(df["p_top.M()"] < 173 + 40) & (df["p_top.M()"] > 173 - 40)]
            # Remove outliers from all columns
            # TODO: is 4000 a good value for all observables
            df = df[df.lt(4000).all(axis=1)]
            # Select relevant columns for GNN
            df = df[hetero_clms]

            # uniform distribution for the background
            df["alpha"] = np.random.choice(parameters, df.shape[0])
            df["label"] = np.zeros(df.shape[0])
            # append to the overall dataframe
            result.append(df)

    # Combine the signal and the background
    df = pd.concat(result, axis=0)
    return df


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Process kinematics CSV file.")
parser.add_argument(
    "--alphas",
    type=list_of_strings,
    help="A list of alpha values to train/test on.",
    required=False,
    default=[
        "0.000000",
        "0.523599",
        "1.047200",
        "1.570800",
        "x.xxxxxx",
    ],
)
parser.add_argument(
    "--file_dir",
    type=str,
    help="Directory to where the file is.",
    required=False,
    default="/home/paperspace/CP-Higgs/files/",
)
parser.add_argument(
    "--frac",
    type=float,
    help="Fraction of selected signal events.",
    required=False,
    default=0.5,
)
args = parser.parse_args()

df = read_clean(root_dir=args.file_dir, alphas=args.alphas, frac=args.frac)

# Initialize an empty list to store the feature vectors
feature_vectors = []
for row in tqdm(df.to_dict("records"), total=len(df), desc="Processing Events "):

    lepton1 = [
        1,  # I1: is lepton
        0,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_l.first.Pt()"],
        row["p_l.first.E()"],
        row["p_l.first.Eta()"],
        row["p_l.first.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    lepton2 = [
        1,  # I1: is lepton
        0,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_l.second.Pt()"],
        row["p_l.second.E()"],
        row["p_l.second.Eta()"],
        row["p_l.second.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    bjet1 = [
        0,  # I1: is lepton
        1,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_b1.Pt()"],
        row["p_b1.E()"],
        row["p_b1.Eta()"],
        row["p_b1.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    bjet2 = [
        0,  # I1: is lepton
        1,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_b2.Pt()"],
        row["p_b2.E()"],
        row["p_b2.Eta()"],
        row["p_b2.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    bjet3 = [
        0,  # I1: is lepton
        1,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["pb_fromHiggs_1.Pt()"],
        row["pb_fromHiggs_1.E()"],
        row["pb_fromHiggs_1.Eta()"],
        row["pb_fromHiggs_1.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    bjet4 = [
        0,  # I1: is lepton
        1,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["pb_fromHiggs_2.Pt()"],
        row["pb_fromHiggs_2.E()"],
        row["pb_fromHiggs_2.Eta()"],
        row["pb_fromHiggs_2.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    met1 = [
        0,  # I1: is lepton
        0,  # I2: is b-jet
        1,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_v1.Pt()"],
        row["p_v1.E()"],
        row["p_v1.Eta()"],
        row["p_v1.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    met2 = [
        0,  # I1: is lepton
        0,  # I2: is b-jet
        1,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["p_v2.Pt()"],
        row["p_v2.E()"],
        row["p_v2.Eta()"],
        row["p_v2.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    top1 = [
        0,  # I1: is lepton
        0,  # I2: is b-jet
        0,  # I3: is neutrino
        1,  # I4: is reconstructed top-quark
        row["p_top.Pt()"],
        row["p_top.E()"],
        row["p_top.Eta()"],
        row["p_top.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    top2 = [
        0,  # I1: is lepton
        0,  # I2: is b-jet
        0,  # I3: is neutrino
        1,  # I4: is reconstructed top-quark
        row["p_tbar.Pt()"],
        row["p_tbar.E()"],
        row["p_tbar.Eta()"],
        row["p_tbar.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    higgs = [
        0,  # I1: is lepton
        0,  # I2: is b-jet
        0,  # I3: is neutrino
        0,  # I4: is reconstructed top-quark
        row["pHiggs.Pt()"],
        row["pHiggs.E()"],
        row["pHiggs.Eta()"],
        row["pHiggs.Phi()"],
        row["Cos_thetahel1"],
        row["Cos_thetahel2"],
        row["Cos_opening"],
        row["Cos_thetaS"],
    ]

    event = [
        row["alpha"],  # 0
        row["label"],  # 1
        row["dPhill"],  # 2
        row["dPhi_lltt"],  # 3
        row["pHiggs.Pt()"],  # 4
        row["pHiggs.E()"],  # 5
        row["pHiggs.Eta()"],  # 6
        row["pHiggs.Phi()"],  # 7
        row["Cos_thetahel1"],  # 8
        row["Cos_thetahel2"],  # 9
        row["Cos_opening"],  # 10
        row["Cos_thetaS"],  # 11
    ]

    event_feature_vector = [
        lepton1,
        lepton2,
        bjet1,
        bjet2,
        bjet3,
        bjet4,
        met1,
        met2,
        top1,
        top2,
        higgs,
        event,
    ]
    feature_vectors.append(event_feature_vector)

features = np.stack(feature_vectors)
features = features.reshape(features.shape[0], -1)

# Convert the list of feature vectors into a DataFrame
feature_df = pd.DataFrame(
    features,
    columns=[
        "Lepton1-I1",
        "Lepton1-I2",
        "Lepton1-I3",
        "Lepton1-I4",
        "Lepton1-pT",
        "Lepton1-E",
        "Lepton1-Eta",
        "Lepton1-Phi",
        "Lepton1-Cos_thetahel1",
        "Lepton1-Cos_thetahel2",
        "Lepton1-Cos_opening",
        "Lepton1-Cos_thetaS",
        "Lepton2-I1",
        "Lepton2-I2",
        "Lepton2-I3",
        "Lepton2-I4",
        "Lepton2-pT",
        "Lepton2-E",
        "Lepton2-Eta",
        "Lepton2-Phi",
        "Lepton2-Cos_thetahel1",
        "Lepton2-Cos_thetahel2",
        "Lepton2-Cos_opening",
        "Lepton2-Cos_thetaS",
        "BJet1-I1",
        "BJet1-I2",
        "BJet1-I3",
        "BJet1-I4",
        "BJet1-pT",
        "BJet1-E",
        "BJet1-Eta",
        "BJet1-Phi",
        "BJet1-Cos_thetahel1",
        "BJet1-Cos_thetahel2",
        "BJet1-Cos_opening",
        "BJet1-Cos_thetaS",
        "BJet2-I1",
        "BJet2-I2",
        "BJet2-I3",
        "BJet2-I4",
        "BJet2-pT",
        "BJet2-E",
        "BJet2-Eta",
        "BJet2-Phi",
        "BJet2-Cos_thetahel1",
        "BJet2-Cos_thetahel2",
        "BJet2-Cos_opening",
        "BJet2-Cos_thetaS",
        "BJet3-I1",
        "BJet3-I2",
        "BJet3-I3",
        "BJet3-I4",
        "BJet3-pT",
        "BJet3-E",
        "BJet3-Eta",
        "BJet3-Phi",
        "BJet3-Cos_thetahel1",
        "BJet3-Cos_thetahel2",
        "BJet3-Cos_opening",
        "BJet3-Cos_thetaS",
        "BJet4-I1",
        "BJet4-I2",
        "BJet4-I3",
        "BJet4-I4",
        "BJet4-pT",
        "BJet4-E",
        "BJet4-Eta",
        "BJet4-Phi",
        "BJet4-Cos_thetahel1",
        "BJet4-Cos_thetahel2",
        "BJet4-Cos_opening",
        "BJet4-Cos_thetaS",
        "MET1-I1",
        "MET1-I2",
        "MET1-I3",
        "MET1-I4",
        "MET1-pT",
        "MET1-E",
        "MET1-Eta",
        "MET1-Phi",
        "MET1-Cos_thetahel1",
        "MET1-Cos_thetahel2",
        "MET1-Cos_opening",
        "MET1-Cos_thetaS",
        "MET2-I1",
        "MET2-I2",
        "MET2-I3",
        "MET2-I4",
        "MET2-pT",
        "MET2-E",
        "MET2-Eta",
        "MET2-Phi",
        "MET2-Cos_thetahel1",
        "MET2-Cos_thetahel2",
        "MET2-Cos_opening",
        "MET2-Cos_thetaS",
        "Top1-I1",
        "Top1-I2",
        "Top1-I3",
        "Top1-I4",
        "Top1-pT",
        "Top1-E",
        "Top1-Eta",
        "Top1-Phi",
        "Top1-Cos_thetahel1",
        "Top1-Cos_thetahel2",
        "Top1-Cos_opening",
        "Top1-Cos_thetaS",
        "Top2-I1",
        "Top2-I2",
        "Top2-I3",
        "Top2-I4",
        "Top2-pT",
        "Top2-E",
        "Top2-Eta",
        "Top2-Phi",
        "Top2-Cos_thetahel1",
        "Top2-Cos_thetahel2",
        "Top2-Cos_opening",
        "Top2-Cos_thetaS",
        "Higgs-I1",
        "Higgs-I2",
        "Higgs-I3",
        "Higgs-I4",
        "Higgs-pT",
        "Higgs-E",
        "Higgs-Eta",
        "Higgs-Phi",
        "Higgs-Cos_thetahel1",
        "Higgs-Cos_thetahel2",
        "Higgs-Cos_opening",
        "Higgs-Cos_thetaS",
        "Event-alpha",
        "label",
        "Event-I3",
        "Event-I4",
        "Event-pT",
        "Event-E",
        "Event-Eta",
        "Event-Phi",
        "Event-Cos_thetahel1",
        "Event-Cos_thetahel2",
        "Event-Cos_opening",
        "Event-Cos_thetaS",
    ],
)

# Save the DataFrame to a CSV file
output_file = f"inputgraphs.pkl"
if len(args.alphas) == 2:
    output_file = f"inputgraphs_{args.alphas[0]}.pkl"
print(f"saving events to {output_file}")
feature_df.to_pickle(args.file_dir + output_file)
