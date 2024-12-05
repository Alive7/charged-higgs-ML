import numpy as np
import matplotlib.pyplot as plt
import vector
import awkward as ak
import uproot as up

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc

import ROOT

import pandas as pd
import joblib

def fill_histogram(data, start, stop, N_bins=80, weights=None, normalize=False, uid: int=0, title: str='hist', x_title: str=None, y_title: str=None):
    ROOT.gROOT.ForceStyle()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetStyle("ATLAS")
    h = ROOT.TH1D("h"+str(uid), title, N_bins, start, stop)
    data_to_fill = data.copy().astype('float64') 
    if x_title is not None:
        h.GetXaxis().SetTitle(x_title)
    if y_title is not None:
        h.GetYaxis().SetTitle(y_title)
    if weights is None:
        weights = np.ones(data_to_fill.size)
    else:
        weights = weights.copy().astype('float64')
    h.FillN(data_to_fill.size,data_to_fill,weights)
    #h.SetLineColor(uid+2)
    #h.Sumw2()
    if normalize:
        h.Scale(1.0/h.Integral(), "width")
    return h

def plot_histograms(historgrams, title: str, extension: str=".pdf", add_ylable: bool=True, add_legend: bool=True, log_scale: bool=False):
    c = ROOT.TCanvas("c", title, 500, 500)
    # use stack instead?
    legend = ROOT.TLegend(0.65, 0.75, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    for i,h in enumerate(historgrams):
        if not i:
            width = h.GetBinWidth(1)
            str_width = str(width)
            if add_ylable:
                h.GetYaxis().SetTitle("Events/"+str_width[:str_width.find('.')+3]+" GeV")
            h.Draw("E1 PLC PMC")
        else:
            h.Draw("E1 SAME PLC PMC")
        legend.AddEntry(h.GetName(), h.GetTitle(), "L")
    legend.SetFillStyle(0)
    if add_legend:
        legend.Draw()
    if log_scale:
        c.SetLogy(1)
    #tex = ROOT.TLatex()
    #tex.SetNDC()
    #tex.SetTextSize(0.035)
    #tex.DrawLatex(0.2+0.02,0.88, "#bf{#it{ATLAS}} WIP")
    #tex.DrawLatex(0.2+0.02,0.85, "DiJet Samples")
    c.Print("plots/"+title+extension)

def plot_BDT_input_features_np(features_data, features_background, feature_names, axis_names, signal_weights = None, background_weights = None):
    mins_feature = np.amin(features_data, axis=0)
    maxs_feature = np.amax(features_data, axis=0)
    
    mins_background = np.amin(features_background, axis=0)
    maxs_background = np.amax(features_background, axis=0)

    mins_plot = np.where(mins_feature < mins_background, mins_feature, mins_background)
    maxs_plot = np.where(maxs_feature > maxs_background, maxs_feature, maxs_background)
    
    #if mins_plot[0] > 0:
    #    mins_plot[0] = 0
    #if mins_plot[1] > 0:
    #    mins_plot[1] = 0
    if maxs_plot[1] > 180:
        maxs_plot[1] = 180
    if maxs_plot[2] > 400:
        maxs_plot[2] = 400
    if maxs_plot[3] > 350:
        maxs_plot[3] = 350
    if maxs_plot[4] > 350:
        maxs_plot[4] = 350
    if maxs_plot[5] > 50:
        maxs_plot[5] = 50
    if maxs_plot[6] > 50:
        maxs_plot[6] = 50
    if maxs_plot[7] > 3.5:
        maxs_plot[7] = 3.5
    if maxs_plot[8] > 3:
        maxs_plot[8] = 3
    if maxs_plot[9] > 3:
        maxs_plot[9] = 3

    for i,name in enumerate(axis_names):
        h_data = fill_histogram(features_data[:,i], mins_plot[i], maxs_plot[i], weights = signal_weights, normalize=True, uid=i+1, title="GM H++: 200GeV", x_title=name)
        h_background = fill_histogram(features_background[:,i], mins_plot[i], maxs_plot[i], weights = background_weights, normalize=True, uid=(i+1)*10, title="background")
        plot_histograms([h_data,h_background], feature_names[i], extension = ".png")

def plot_BDT_input_features_pd(df_sig, df_bkg):
    names = list(df_sig.columns)
    axes = list(df_sig.columns)
    weight_sig = df_sig['weight'].to_numpy()
    weight_bkg = df_bkg['weight'].to_numpy()
    features_sig = df_sig.to_numpy()
    features_bkg = df_bkg.to_numpy()
    for i in range(1,7):
        axes[i] = axes[i] + " (GeV)"
        features_sig[:,i] /= 1000
        features_bkg[:,i] /= 1000
    print(names)

    plot_BDT_input_features_np(features_sig,features_bkg,names,axes,weight_sig,weight_bkg)

def BDT_pipeline():
    files_data, files_background = get_data_paths()

    keys = ['dyjj', 'jet0_pt', 'jet1_pt', 'lep0_px', 'lep0_py', 'lep0_pz', 'lep0_e', 'lep1_px', 'lep1_py', 'lep1_pz', 'lep1_e', 'weight']
    data = up.concatenate(files_data,keys)
    background = up.concatenate(files_background,keys)

    features_data, weight_data = get_BDT_input_features(data)
    features_background, weight_background = get_BDT_input_features(background)
    feature_names = ["pt_jet0", "pt_jet1", "dyjj", "mll", "d_eta_ll", "dyll"]
    #plot_BDT_input_features(features_data, features_background, feature_names, data["weight"].to_numpy(), background["weight"].to_numpy())

    targets_data = np.ones(features_data.shape[0])
    targets_background = np.zeros(features_background.shape[0])

    print(targets_background.size/targets_data.size)

    features = np.vstack((features_data, features_background))
    targets = np.hstack((targets_data, targets_background))
    weights = np.hstack((weight_data, weight_background))

    Cij = np.corrcoef(features.T)
    print(feature_names)
    print(Cij)

    features = features[weights > 0]
    targets = targets[weights > 0]
    weights = weights[weights > 0]

    train_features, test_features, train_targets, test_targets, train_weights, test_weights = train_test_split(features, targets, weights, test_size=.1, random_state=10)
    
    train_features_net_weight = train_weights[train_targets == 1].sum()
    train_background_net_weight = train_weights[train_targets == 0].sum()

    train_weights[train_targets == 1] = train_weights[train_targets == 1] / train_features_net_weight
    train_weights[train_targets == 0] = train_weights[train_targets == 0] / train_background_net_weight

    bdt = ensemble.GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            validation_fraction=0.1, #.11
            n_iter_no_change=10,
            tol=0.01,
            random_state=10,
            verbose=1,
        )

    bdt.fit(train_features, train_targets, sample_weight=train_weights)

    fig, ax = plt.subplots()

    feature_importance = bdt.feature_importances_
    bars = zip(feature_names,feature_importance)
    bars = dict(reversed(sorted(bars, key=lambda bar: bar[1])))

    ax.bar(bars.keys(), bars.values())

    ax.set_ylabel('Feature importance')
    ax.set_title('Feature importances for mass 200 BDT')
    plt.xticks(rotation=45)
    #ax.legend(title='importance')

    plt.savefig("plots/feature_importance.png")
    
    test_outputs = bdt.predict_proba(test_features)[:,1]
    # plot over training
    train_outputs = bdt.predict_proba(train_features)[:,1]

    h_test_bkg = fill_histogram(test_outputs[test_targets == 0], 0, 1, 20, normalize=True, uid=0, title="test bkg")
    h_test_sig = fill_histogram(test_outputs[test_targets == 1], 0, 1, 20, normalize=True, uid=1, title="test sig")
    
    #h_train_bkg = fill_histogram(train_outputs[train_targets == 0], 0, 1, 51, normalize=True, uid=2, title="train bkg")
    #h_train_sig = fill_histogram(train_outputs[train_targets == 1], 0, 1, 51, normalize=True, uid=3, title="train sig")

    hists = [h_test_bkg, h_test_sig]

    plot_histograms(hists, "bdt_signal_background", ".png", add_ylable=False, log_scale=True)
    #plot_histograms(hists[0::2], "bdt_bkg", add_ylable=False)
    #plot_histograms(hists[1::2], "bdt_sig", add_ylable=False)
    
    #plot roc
    fpr, tpr, thresholds = roc_curve(test_targets, test_outputs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (area = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Randomly guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.savefig("plots/roc_curve.png")

################################
################################
################################
def get_dict(array_ak, branches: list, keys: list, to_numpy: bool=False):
    assert(type(branches) is list)
    assert(type(keys) is list)
    if to_numpy:
        data = [array_ak[b].to_numpy() for b in branches]
    else:
        data = [array_ak[b] for b in branches]
    return {key:value for key,value in zip(keys,data)}

def get_BDT_input_features(ntuple_paths: list):
    # more efficent to process a small number of large files than a large number of small files
    assert(type(ntuple_paths) is list)
    # only use WW baseline (for now)
    # look into event_selection_NOSYS. Get inspired by https://gitlab.cern.ch/adscott/sr-optimization/-/tree/master?ref_type=heads
    # further explanations are found at https://gitlab.cern.ch/atlas-physics/hmbs/mbl/ana-hdbs-2022-12/vbstools/-/blob/v03.1/VBSTools/CustomVariables.h?ref_type=tags#L43
    print("WARNING: Only WW baseline selctions are used. Additional selections must be applied to select appropriate jets, leptons, and signal regions.")
    s_baseline = 'pass_WW_baseline_NOSYS'

    # define branches to be read
    b_el = ['el_pt_NOSYS', 'el_eta', 'el_phi', 'el_e_NOSYS']
    b_mu = ['mu_pt_NOSYS', 'mu_eta', 'mu_phi', 'mu_e_NOSYS']
    b_jets = ['jet_pt_NOSYS', 'jet_eta', 'jet_phi', 'jet_e_NOSYS']
    b_met = ['met_met_NOSYS', 'met_phi_NOSYS']
    b_weights = ['weight_mc_NOSYS', 'weight_pileup_NOSYS']#, 'weight_jvt_effSF_NOSYS', 'weight_fjvt_effSF_NOSYS']
    b_features = ['dRap_jj_NOSYS', 'mll_NOSYS', 'MT_dilep_NOSYS', 'jet1_pt_NOSYS', 'jet2_pt_NOSYS']#, 'dPhi_ll_NOSYS', 'pt_ll_NOSYS', 'Et_ll_NOSYS', 'dEta_jj_NOSYS', 'dPhi_jj_NOSYS', 'mjj_NOSYS']
    #b_selections = ['pass_WW_baseline_NOSYS']#, 'el_WW_selected_NOSYS', 'mu_WW_selected_NOSYS']
    branches = b_el + b_mu + b_jets + b_met + b_weights + b_features# + b_selections

    four_components = ['pt', 'eta', 'phi', 'E']
    two_components = ['pt', 'phi']
    feature_names = ['DYjj', 'Mll', 'MT', 'jet0_pt', 'jet1_pt']#, 'jet0_m', 'jet1_m', 'DPhillMET', 'DEtall', 'DYll']
    
    #i = 0
    dfs = []
    # loop over ntuples in well defined memory chunks
    for array in up.iterate(ntuple_paths, branches, s_baseline, step_size="100 MB"):
        four_el = vector.zip(get_dict(array, b_el, four_components))
        four_mu = vector.zip(get_dict(array, b_mu, four_components))
        four_leps = ak.concatenate([four_el,four_mu],axis=1)
        #print(ak.sum(ak.count(four_leps.pt,axis=1) > 2))
        #print(ak.sum(ak.count(four_leps.pt,axis=1) == 1))
        #print(ak.sum(ak.count(four_leps.pt,axis=1) < 1))

        four_jets = vector.zip(get_dict(array, b_jets, four_components))
        two_met = vector.zip(get_dict(array, b_met, two_components))

        four_leps = four_leps[ak.argsort(four_leps.pt, ascending=False)]    
        four_jets = four_jets[ak.argsort(four_jets.pt, ascending=False)]

        lep0 = four_leps[:,0]
        lep1 = four_leps[:,1]
        leps = lep0 + lep1

        jet0 = four_jets[:,0]
        jet1 = four_jets[:,1]

        # compute jet mass since not pre calculated
        jet0_m = jet0.M.to_numpy()
        jet1_m = jet1.M.to_numpy()

        # compute jet pt and dy for comparison
        #jet0_pt = jet0.pt.to_numpy()
        #jet1_pt = jet1.pt.to_numpy()
        #d_y_jj = np.abs(jet0.rapidity - jet1.rapidity).to_numpy()

        # compute non-precomputed lepton input features
        d_eta_ll = np.abs(lep0.deltaeta(lep1)).to_numpy()
        d_y_ll = np.abs(lep0.rapidity-lep1.rapidity).to_numpy()
        d_phi_ll_met = np.abs(leps.phi-two_met.phi).to_numpy()

        # compute remaining lepton input features for comparison
        #mll = leps.M.to_numpy()
        #d_phi_ll = lep1.deltaphi(lep0).to_numpy()
        #pt_ll = leps.pt.to_numpy()

        #leps_transverse = vector.zip({'pt': leps.pt, 'phi': leps.phi})
        #et_ll = np.sqrt(leps.pt*leps.pt+leps.M*leps.M)
        #a = two_met.pt + et_ll
        #b = two_met + leps_transverse
        #MT = np.sqrt(a*a-b.dot(b)).to_numpy()

        # construct dictorary of np arrays to construct pd dataframe
        np_dict = get_dict(array, b_features, feature_names, to_numpy=True)
        np_dict['jet0_m'] = jet0_m
        np_dict['jet1_m'] = jet1_m    
        np_dict['DPhillMET'] = d_phi_ll_met
        np_dict['DEtall'] = d_eta_ll
        np_dict['DYll'] = d_y_ll
        print("WARNING: reco weights are probably not computed correctly. See https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/PMGCrossSectionTool")
        np_dict['weight'] = array[b_weights[0]] * array[b_weights[1]]
        
        df = pd.DataFrame(np_dict)
        dfs.append(df)
        #i += 1
    #print(i)
    df = pd.concat(dfs, ignore_index=True)
    return df

def print_ntuple_branches(file: str):
    with up.open(file) as f:
        print(f.classnames())
        tree = f[f.keys()[-1]]
        keys = []
        for key in sorted(tree.keys(),reverse=False):
            if "NOSYS" in key:
                print(key)
                if "el_" in key:
                    keys.append(key)
                elif "mu_" in key:
                    keys.append(key)
    return keys

def get_ntuple_paths(ntuple_files: list, DSIDs: list):
    if type(DSIDs) is not list:
        assert(type(DSIDs) is str)
        DSIDs = [DSIDs]
    data_paths = []
    for file in ntuple_files:
        with open(file) as f:
            txt = f.read()
        for id in DSIDs:
            search_start = 0
            # find every ntuple associated with a given dsid
            while txt.find(id, search_start) >= 0:
                line_start = txt.find(id, search_start)
                line_end = txt.index('\n',line_start)
                # the structure of the ntuple lists may change in the future
                path = txt[line_start:line_end].split()[-1]
                path = '/'+path.split('//')[-1]
                data_paths.append(path)
                search_start = line_end + 1
    return data_paths

def process_ntuples(DSIDs: list, files: list, sig: bool=False):
    assert(type(DSIDs) is list)
    assert(type(files) is list)
    # can loop over slices of DSIDs if several DSIDs corrspond to one classification of background
    for id in DSIDs:
        ntuple_paths = get_ntuple_paths(files,id)
        if sig:
            #print_ntuple_branches(ntuple_paths[0])
            bdt_inputs_df = get_BDT_input_features(ntuple_paths)
            fname = id+'_sig_merged_SR'
        else:
            bdt_inputs_df = get_BDT_input_features(ntuple_paths[:1])
            fname = id+'_bkg_merged_SR'
        # save dataframes as joblib dumps for compatibility with mc16 analysis
        # this should be scrapped in run 3
        joblib.dump(bdt_inputs_df, 'compressed_dataframes/'+fname+'.gz',compress=("gzip", 3))
        # make dumps of processed ntuples to simplify further analyses
        with up.recreate('processed_ntuples/'+fname+'.root') as f:
            f['defaultTree;1'] = bdt_inputs_df
    return

def load_dfs(DSIDs: list, sig: bool):
    if type(DSIDs) is not list:
        assert(type(DSIDs) is str)
        DSIDs = [DSIDs]
    dfs = []
    for id in DSIDs:
        if sig:
            fname = id+'_sig_merged_SR.gz'
        else:
            fname = id+'_bkg_merged_SR.gz'
        df = joblib.load("compressed_dataframes/"+fname)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df 
        
def main():
    # absolute path to files containing the ntuples for mc20 and mc23
    path = "/eos/atlas/atlascerngroupdisk/phys-hmbs/mbl/ssWWWZ_run3/ntuple_v02/"
    # absolute paths to the lists of ntuples for a given run
    files = ["filelist_mc20a.txt", "filelist_mc20d.txt", "filelist_mc20e.txt"]
    paths = [path + file for file in files]
    # desired sig/bkg DSIDs
    DSIDs_ssWW = ["700590", "700603", "700594"]
    DSIDs_WZ = ["700588", "700601", "700592"]
    DSIDs_sig_450 = ["525935"]
    DSIDs_sig_200 = ["525925"]
    # combine bkgs
    DSIDs_bkgs = DSIDs_ssWW + DSIDs_WZ
    DSIDs_sigs = DSIDs_sig_200
    # process selected signal and backgrounds
    #process_ntuples(DSIDs_bkgs, paths, False)
    #process_ntuples(DSIDs_sigs, paths, True)

    df_sig = load_dfs(DSIDs_sigs, True)
    df_bkg = load_dfs(DSIDs_bkgs, False)
    plot_BDT_input_features_pd(df_sig,df_bkg)
    
if __name__ == "__main__":
    main()    
