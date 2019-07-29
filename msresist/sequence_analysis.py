from Bio import SeqIO
import os
import numpy as np
import random


def GenerateFastaFile(PathToFile, PN, X_seqs):
    """ Sequence processor """
    FileHandle = open(PathToFile, "w+")
    for i in range(len(X_seqs)):
        FileHandle.write('>' + str(PN[i]))
        FileHandle.write("\n")
        FileHandle.write(str(X_seqs[i]))
        FileHandle.write("\n")
    FileHandle.close()


# def YTSsequences(X_seqs):
#    """ Dictionary to Check Motifs
#        Input: Phosphopeptide sequences
#        Output: Dictionary to see all sequences categorized by singly or doubly phosphorylated.
#        Useful to check def GeneratingKinaseMotifs results """
#    YTSsequences = {}
#    seq1 , seq2, seq3, seq4, seq5, seq6, = [], [], [], [], [], []
#    for i, seq in enumerate(X_seqs):
#        if "y" in seq and "t" not in seq and "s" not in seq:
#            seq1.append(seq)
#        if "t" in seq and "y" not in seq and "s" not in seq:
#            seq2.append(seq)
#            DictProtNameToPhospho["t: "] = seq2
#        if "s" in seq and "y" not in seq and "t" not in seq:
#            seq3.append(seq)
#            DictProtNameToPhospho["s: "] = seq3
#        if "y" in seq and "t" in seq and "s" not in seq:
#            seq4.append(seq)
#            DictProtNameToPhospho["y/t: "] = seq4
#        if "y" in seq and "s" in seq and "t" not in seq:
#            seq5.append(seq)
#            DictProtNameToPhospho["y/s: "] = seq5
#        if "t" in seq and "s" in seq and "y" not in seq:
#            seq6.append(seq)
#
#    DictProtNameToPhospho["y: "] = seq1
#    DictProtNameToPhospho["t: "] = seq2
#    DictProtNameToPhospho["s: "] = seq3
#    DictProtNameToPhospho["y/t: "] = seq4
#    DictProtNameToPhospho["y/s: "] = seq5
#    DictProtNameToPhospho["t/s: "] = seq6
#
#    SeqsBySites = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in DictProtNameToPhospho.items() ]))
#
#    return SeqsBySites


###------------ Match protein names from MS to Uniprot's data set ------------------###

"""Input: Path to new file and MS fasta file
   Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
   Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
"""


# def MatchProtNames(PathToNewFile, MS_seqs):
##     FileHandle = open("./msresist/data/MS_seqs_matched.fa", "w+")
#    FileHandle = open(PathToNewFile, "w+")
#    # counter = 0
#    for rec1 in SeqIO.parse(MS_seqs, "fasta"):
#        MS_seq = str(rec1.seq)
#        MS_seqU = str(rec1.seq.upper())
#        MS_name = str(rec1.description.split(" OS")[0])
#        try:
#            UP_seq = DictProtToSeq_UP[MS_name]
#            FileHandle.write(">" + MS_name)
#            FileHandle.write("\n")
#            FileHandle.write(MS_seq)
#            FileHandle.write("\n")
#        except:
#            # counter += 1
#            Fixed_name = getKeysByValue(DictProtToSeq_UP, MS_seqU)
#            FileHandle.write(">" + Fixed_name[0])
#            FileHandle.write("\n")
#            FileHandle.write(MS_seq)
#            FileHandle.write("\n")
#    FileHandle.close()


###------------ Generate Phosphopeptide Motifs ------------------###

"""Input: Fasta file and Uniprot's proteome dictionary key: Protein accession value: protein sequence
   Output: Protein names list and kinase motif list. Run with def GenerateFastaFile to obtain the final file
   Kinase motif -5 +5 wrt the phosphorylation site. It accounts for doubly phosphorylated peptides (lowercase y, t, s)
"""


# def GeneratingKinaseMotifs(PathToFile, DictProtToSeq_UP):
#    ExtSeqs = []
#    MS_names = []
#    for rec1 in SeqIO.parse(MS_seqs_matched, "fasta"):
#        MS_seq = str(rec1.seq)
#        MS_seqU = str(rec1.seq.upper())
#        MS_name = str(rec1.description.split(" OS")[0])
#        MS_names.append(MS_name)
#        try:
#            UP_seq = DictProtToSeq_UP[MS_name]
#            if MS_seqU in UP_seq and MS_name == list(DictProtToSeq_UP.keys())[list(DictProtToSeq_UP.values()).index(UP_seq)]:
#                counter += 1
#                regexPattern = re.compile(MS_seqU)
#                MatchObs = regexPattern.finditer(UP_seq)
#                indices = []
#                for i in MatchObs:
#                    indices.append(i.start())  # VHLENATEYAtLR   #YNIANtV
#                    indices.append(i.end())
#                if "y" in MS_seq and "t" not in MS_seq and "s" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeqs.append(UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6])
#
#                if "t" in MS_seq and "y" not in MS_seq and "s" not in MS_seq:
#                    t_idx = MS_seq.index("t") + indices[0]
#                    ExtSeqs.append(UP_seq[t_idx - 5:t_idx] + "t" + UP_seq[t_idx + 1:t_idx + 6])
#
#                if "s" in MS_seq and "y" not in MS_seq and "t" not in MS_seq:
#                    s_idx = MS_seq.index("s") + indices[0]
#                    ExtSeqs.append(UP_seq[s_idx - 5:s_idx] + "s" + UP_seq[s_idx + 1:s_idx + 6])
#
#                if "y" in MS_seq and "t" in MS_seq and "s" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeq = UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6]
#                    y_idx = MS_seq.index("y")
#                    if "t" in MS_seq[y_idx - 5:y_idx + 6]:
#                        t_idx = MS_seq[y_idx - 5:y_idx + 6].index("t")
#                        ExtSeqs.append(ExtSeq[:t_idx] + "t" + ExtSeq[t_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#
#                if "y" in MS_seq and "s" in MS_seq and "t" not in MS_seq:
#                    y_idx = MS_seq.index("y") + indices[0]
#                    ExtSeq = UP_seq[y_idx - 5:y_idx] + "y" + UP_seq[y_idx + 1:y_idx + 6]
#                    y_idx = MS_seq.index("y")
#                    if "s" in MS_seq[y_idx - 5:y_idx + 6]:
#                        s_idx = MS_seq[y_idx - 5:y_idx + 6].index("s")
#                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#
#                if "t" in MS_seq and "s" in MS_seq and "y" not in MS_seq:
#                    t_idx = MS_seq.index("t") + indices[0]
#                    ExtSeq = UP_seq[t_idx - 5:t_idx] + "t" + UP_seq[t_idx + 1:t_idx + 6]
#                    t_idx = MS_seq.index("t")
#                    if "s" in MS_seq[t_idx - 5:t_idx + 6]:
#                        s_idx = MS_seq[t_idx - 5:t_idx + 6].index("s")
#                        ExtSeqs.append(ExtSeq[:s_idx] + "s" + ExtSeq[s_idx + 1:])
#                    else:
#                        ExtSeqs.append(ExtSeq)
#        except BaseException:
#            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
#            pass
#
#        return MS_names, ExtSeqs

###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

# Code from Adam Weiner, obtained March 2019


def trim(seqFile):
    cwd = os.getcwd()
    homeDir = cwd[1:5]
    if (homeDir == 'home'):
        print('using path from server to load sequences')
#         pathToFile = os.path.join("/home","zoekim","Desktop",str(seqFile)) #aretha server
        pathToFile = os.path.join("/home", "marcc", "resistance-MS", "msresist", "data", str(seqFile))  # /home/marcc/resistance-MS/msresist/data

    else:
        print('using path from mac machine to load sequences')
        pathToFile = os.path.join("/Users", "zoekim", "Desktop", str(seqFile))  # mac machine

    allSeqs = []
    allLabels = []
    for seq_record in SeqIO.parse(pathToFile, """fasta"""):
        allSeqs.append(seq_record.seq)
        allLabels.append(seq_record.id)

    seqMat = np.array(allSeqs)
    label = np.array(allLabels)

    sequence = seqMat[:, 0:317]

    # filtering out residues not included in PAM250 pymsa distance matrix (http://www.matrixscience.com/blog/non-standard-amino-acid-residues.html)
    for i in range(0, sequence.shape[0]):
        for j in range(0, sequence.shape[1]):
            if (sequence[i, j] == 'J'):
                sequence[i, j] = random.choice(['I', 'L'])
    print(label)
    print(sequence)

    return (label, sequence)


class Distance:
    """ Seq Distance Calculator
        Code from Adam Weiner, obtained March 2019 """

    def __init__(self, seqFile, subMat):
        self.labels, self.sequences = trim(seqFile)
        self.subMat = subMat
        if subMat is "PAM250":
            self.M = PAM250()
        self.numSeq = self.sequences.shape[0]

    def seq_dist(self, seq1, seq2):
        if (len(seq1) != len(seq2)):
            print('the sequences are of different length')
            return -1
        else:
            dist = np.zeros((len(seq1)))
            for ii in range(len(seq1)):
                temp_dist = self.M.get_distance(seq1[ii], seq2[ii])
                if self.subMat is "PAM250":  # convert log-scaled PAM250 values to true values
                    temp_dist = np.exp(temp_dist)
                dist[ii] = 1 / temp_dist  # large distances have small values in matrices
#             avg_dist = np.sum(dist) / 317.0

            return np.sum(dist)

    def test_mat(self):
        """ function is the same as "dist_mat()" except that it only looks at first 10 sequences
        in order to get a proof of concept for all my functions before scaling up to the full dataset
        """
        testMat = np.zeros((1000, 1000))
        print('calculating the test distance matrix based on PAM250')
        for i in range(0, 1000):
            for j in range(i, 1000):
                testMat[i, j] = self.seq_dist(self.sequences[i], self.sequences[j])
                # plug in values for mirror images
                testMat[j, i] = testMat[i, j]

        return testMat

    def dist_mat(self):
        distMat = np.zeros((self.numSeq, self.numSeq))
        print(self.numSeq)
        print(self.sequences.shape)
        print('calculating the full distance matrix based on PAM250')
        for i in range(0, self.numSeq):
            for j in range(i, self.numSeq):
                distMat[i, j] = self.seq_dist(self.sequences[i], self.sequences[j])
                distMat[j, i] = distMat[i, j]  # plug in mirror image values
        return distMat


###------------ Substitution Matrix (PAM250) ------------------###
# Code from Adam Weiner, obtained March 2019

# Just load distance matrices from pyMSA!
