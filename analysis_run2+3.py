import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import vector
import awkward as ak
import uproot as up

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc

import ROOT

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

def plot_BDT_input_features(features_data, features_background, feature_names, axis_names, signal_weights = None, background_weights = None):
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
    #if maxs_plot[0] > 350:
    #    maxs_plot[0] = 350
    #if maxs_plot[1] > 350:
    #    maxs_plot[1] = 350
    #if maxs_plot[3] > 180:
    #    maxs_plot[3] = 180

    for i,name in enumerate(axis_names):
        h_data = fill_histogram(features_data[:,i], mins_plot[i], maxs_plot[i], weights = signal_weights, normalize=True, uid=i+1, title="GM H++: 200GeV", x_title=name)
        h_background = fill_histogram(features_background[:,i], mins_plot[i], maxs_plot[i], weights = background_weights, normalize=True, uid=(i+1)*10, title="background")
        plot_histograms([h_data,h_background], feature_names[i], extension = ".png")

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
def get_BDT_input_features(ntuple_paths: list):
    selections = 'pass_WW_baseline_NOSYS'
    
    branches_el = ['pt', 'eta', 'phi', 'E']
    aliases_el = {'n':'el_n_NOSYS', 'eta':'el_eta', 'phi':'el_phi', 'pt': 'el_pt_NOSYS', 'E': 'el_e_NOSYS'}
    data_el = up.concatenate(ntuple_paths, branches_el, selections, aliases=aliases_el)

    branches_mu = ['pt', 'eta', 'phi', 'E']
    aliases_mu = {'n':'mu_n_NOSYS', 'eta':'mu_eta', 'phi':'mu_phi', 'pt': 'mu_pt_NOSYS', 'E': 'mu_e_NOSYS'}
    data_mu = up.concatenate(ntuple_paths, branches_mu, selections, aliases=aliases_mu)

    branches_jets = ['n', 'pt', 'eta', 'phi', 'E']
    aliases_jets = {'n':'jet_n_NOSYS', 'eta':'jet_eta', 'phi':'jet_phi', 'pt': 'jet_pt_NOSYS', 'E': 'jet_e_NOSYS'}
    data_jets = up.concatenate(ntuple_paths, branches_jets, selections, aliases=aliases_jets)

    branches_met = ['met', 'phi']
    aliases_met = {'met':'met_met_NOSYS', 'phi':'met_phi_NOSYS'}
    data_met = up.concatenate(ntuple_paths, branches_met, selections, aliases=aliases_met)

    #branches_features = ['pt_ll_NOSYS', 'mll_NOSYS', 'mjj_NOSYS', 'Et_ll_NOSYS', 'dEta_jj_NOSYS', 'dPhi_jj_NOSYS', 'dPhi_ll_NOSYS', 'dRap_jj_NOSYS', 'MT_dilep_NOSYS']
    branches_features = ['d_y_jj', 'mll', 'd_phi_ll', 'MT', 'pt_ll', 'et_ll']
    aliases_features = {'d_y_jj':'dRap_jj_NOSYS', 'mll':'mll_NOSYS', 'd_phi_ll':'dPhi_ll_NOSYS', 'MT':'MT_dilep_NOSYS', 'pt_ll':'pt_ll_NOSYS', 'et_ll':'Et_ll_NOSYS'}
    data_features = up.concatenate(ntuple_paths, branches_features, selections, aliases=aliases_features, library="np")
    
    branches_weights = ['weight_mc', 'weight_pileup']
    aliases_weights = {'weight_mc':'weight_mc_NOSYS', 'weight_pileup':'weight_pileup_NOSYS'}
    data_weights = up.concatenate(ntuple_paths, branches_weights, selections, aliases=aliases_weights)

    data_leps = ak.concatenate([data_el,data_mu],axis=1)
    leps = vector.zip({'pt': data_leps.pt, 'eta': data_leps.eta, 'phi': data_leps.phi, 'E': data_leps.E})
    jets = vector.zip({'pt': data_jets.pt, 'eta': data_jets.eta, 'phi': data_jets.phi, 'E': data_jets.E})

    leps = leps[ak.argsort(leps.pt, ascending=False)]    
    jets = jets[ak.argsort(jets.pt, ascending=False)]

    lep0 = leps[:,0]
    lep1 = leps[:,1]
    lep_sum = ak.sum(leps, axis=1)
    
    jet0 = jets[:,0]
    jet1 = jets[:,1]    

    jet0_m = jet0.M.to_numpy()
    jet0_pt = jet0.pt.to_numpy()

    jet1_m = jet1.M.to_numpy()
    jet1_pt = jet1.pt.to_numpy()

    d_y_jj = np.abs(jet0.rapidity - jet1.rapidity).to_numpy()

    mll = lep_sum.M.to_numpy()
    d_eta_ll = np.abs(lep0.deltaeta(lep1)).to_numpy()
    d_y_ll = np.abs(lep0.rapidity-lep1.rapidity).to_numpy()

    d_phi_ll = lep0.deltaphi(lep1).to_numpy()
    d_phi_ll_met = np.abs(lep_sum.phi-data_met.phi).to_numpy()

    leps_transverse = vector.zip({'pt': lep_sum.pt, 'phi': lep_sum.phi})
    met = vector.zip({'pt': data_met.met, 'phi': data_met.phi})
    a = met.pt + lep_sum.pt*lep_sum.E/np.sqrt(lep_sum.E*lep_sum.E-lep_sum.M*lep_sum.M)
    b = met + leps_transverse
    MT = np.sqrt(a*a-b.dot(b)).to_numpy()
    pt_ll = lep_sum.pt.to_numpy()
    et_ll = (lep_sum.pt * lep_sum.E/np.sqrt(lep_sum.E*lep_sum.E-lep_sum.M*lep_sum.M)).to_numpy()

    sigs = np.column_stack((d_y_jj,mll,d_phi_ll,MT,pt_ll,et_ll))
    bkgs = np.column_stack(list(data_features.values()))

    names = ['d_y_jj', 'mll', 'd_phi_ll', 'MT', 'pt_ll', 'et_ll']
    plot_BDT_input_features(sigs,bkgs,names,names)

    #axis_names = ["jet 0 m (GeV)", "jet 0 pt (GeV)", "jet 1 m (GeV)", "jet 1 pt (GeV)", "#Delta y_{jj}", "m_{ll} (GeV)", "#Delta #eta_{ll}", "#Delta y_{ll}", "#Delta#phi_{ll#text{-met}}", "MT (GeV)"]
    #feature_names = ['jet0_m','jet0_pt','jet1_m','jet1_pt','d_y_jj','mll','d_eta_ll','d_y_ll','d_phi_ll_met','MT']
    #BDT_input_features = np.column_stack((jet0_m,jet0_pt,jet1_m,jet1_pt,d_y_jj,mll,d_eta_ll,d_y_ll,d_phi_ll_met,MT))
    #backgrounds = np.zeros_like(BDT_input_features)
    #backgrounds[:,4]+=data_features['d_eta_jj']
    #backgrounds[:,5]+=data_features['mll']
    #backgrounds[:,9]+=data_features['MT']
    #print(BDT_input_features.shape)
    #print(data_features['mll'].shape)
    #plot_BDT_input_features(BDT_input_features,backgrounds,feature_names,axis_names)

    return

def get_ntuple_paths(ntuple_list_files: list, DSIDs: list):
    data_paths = []
    for file in ntuple_list_files:
        with open(file) as f:
            txt = f.read()
        for id in DSIDs:
            search_start = 0
            while txt.find(id, search_start) >= 0:
                line_start = txt.find(id, search_start)
                line_end = txt.index('\n',line_start)
                path = txt[line_start:line_end].split()[-1]
                path = '/'+path.split('//')[-1]
                data_paths.append(path)
                search_start = line_end + 1
    return data_paths

def print_ttree_branches(ttree: str):
    with up.open(ttree) as f:
        print(f.classnames())
        tree = f[f.keys()[-1]]
        for key in sorted(tree.keys(),reverse=False):
            print(key)

def create_reduced_ttree(ttree: str):
    with up.open(ttree) as f:
        print(f.classnames())
        tree = f[f.keys()[-1]]
        for key in sorted(tree.keys(),reverse=False):
            if "NOSYS" in key:
                print(key)

def main():
    # everything depends on data set: mc16, mc20, mc23
    path = "/eos/atlas/atlascerngroupdisk/phys-hmbs/mbl/ssWWWZ_run3/ntuple_v02/"
    ntuple_mc20_lists = [path + "filelist_mc20a.txt", path + "filelist_mc20d.txt", path + "filelist_mc20e.txt"]
    
    #DSIDs_mc20_ssWW = ["700590", "700603", "700594"]
    #DSIDs_mc20_WZ = ["700588", "700601", "700592"]
    DSIDs_mc20_sig = ["525925", "525940", "525945"]
    
    #ntuple_paths_ssWW = get_ntuple_paths(ntuple_mc20_lists,DSIDs_mc20_ssWW)
    #ntuple_paths_WZ = get_ntuple_paths(ntuple_mc20_lists,DSIDs_mc20_WZ)
    ntuple_paths_sig = get_ntuple_paths(ntuple_mc20_lists,DSIDs_mc20_sig)
    #print(ntuple_paths_sig)
    #create_reduced_ttree(ntuple_paths_sig[0])

    #bdt_input_features_ssWW = get_BDT_input_features(ntuple_paths_ssWW)
    #bdt_input_features_WZ = get_BDT_input_features(ntuple_paths_WZ)
    bdt_input_features_sig = get_BDT_input_features(ntuple_paths_sig[:2])
    
if __name__ == "__main__":
    main()    
