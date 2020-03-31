import glob
import nltk.corpus
from nltk.corpus import wordnet as wn 
import os
import sys
import getopt
import codecs
import re
from collections import OrderedDict
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')


sys.setrecursionlimit(10000)
list_x = ["abstract", "abstraction", "abstracts", "abstracting", "abstractive", "abstracted", "abstractedness", "acquire", "acquired", "acquires", "acquiring", "acquisition", "acquisitions", "apprehended", "apprehending", "apprehends", "apprehension", "aware", "awareness", "belief", "beliefs", "cognition", "cognitive", "cognitively", "cognizance", "cognizant", "cognized", "concept", "concepts", "conceptual", "conception", "conceptions", "conceptualization", "conceptually", "conceived", "conceivable", "conceive", "conceivably", "conceiving", "conceivability", "conceives", "epistemology", "epistemological", "epistemologically", "epistemologist", "epistemologists", "epistemic", "epistemically", "epistemologist's", "evidence", "evident", "evidential", "evidentially", "evidences", "grasp", "grasped", "grasps", "grasping", "idea", "ideas", "imagine", "imagined", "imagination", "imagining", "imaginary", "imaginative", "imagines", "imaginable", "imaginations", "insight", "insights", "insightful", "justification", "justify", "justified", "justifiable", "justifying", "justifies", "justificatory", "justifications", "know", "known", "knowledge", "knowing", "knows", "knowingly", "knowable", "knowledgeable", "learn", "learning", "learned", "learns", "learner", "learnable", "learner's", "learners", "mental", "mentalistic", "mentalism", "mentalistically", "mentalists", "mentalist", "mind", "minds", "perception", "perceptual", "perceptually", "perceptions", "perceived", "perceives", "perceiving", "perceptible", "percepts", "perceptibly", "perceptive", "sceptic", "scepticism", "sceptical", "sceptic's", "sceptics", "skeptic", "skepticism", "skeptical", "skeptic's", "skeptics", "think", "thinking", "thinks", "thinker", "thinkers", "thought", "thoughts", "truth", "truths", "truthfulness", "understanding", "understand", "understands", "understandable", "understandably", "understandings", "warrant", "warranted", "warrantedness", "warrants", "world", "worlds"]
list_y = ["anti-scientific", "cartesian", "clairvoyance", "descartes", "extra-sensory", "extrasensory", "fiction", "fictions", "fictitious", "fictional", "fictive", "fictionalized", "fictioneer", "foundations", "foundation", "foundational", "foundationalism", "innate", "innately", "innateness", "metaphysics", "metaphysical", "metaphysicians", "metaphysician", "metaphysically", "phenomena", "phenomenon", "phenomenalistic", "phenomenalism", "phenomenology", "phenomenal", "phenomenalist", "phenomenological", "phenomen", "phenomenologically", "rational", "reconstruction", "religion", "religions", "religious", "revelation", "sense", "data", "datum", "supra-scientific", "telepathy", "telepathic", "telepaths", "transcendent", "transcendental", "transcendence", "transcends", "transcend", "transcended", "transcending", "transcendentals", "transcendentally", "transcendently", "unscientific", "account", "accounted", "accounts", "accounting", "accountable", "accountability", "biology", "biological", "biologist", "biologically", "biologists", "biologizing", "black box", "black", "box", "build", "building", "builds", "builder", "carnap", "carnap's", "carnapian", "common sense", "commonsense", "commonsensical", "construction", "constructions", "construct", "constructed", "constructive", "constructible", "constructing", "constructs", "constructivism", "constructivity", "constructibility", "constructional", "ethology", "ethological", "ethologists", "empirical", "empiricism", "empirically", "empiricist", "empiricists", "empiricistic", "empiricist's", "empiricolinguistic", "genetic", "genetically", "genetics", "history", "histories", "historical", "historically", "historicist", "historico", "historian", "historians", "historiography", "hume", "hume's", "humean", "humes", "impact", "impacts", "impingements", "impingement", "impinges", "impinging", "impinge", "input", "inputs", "intake", "intakes", "irradiation", "irradiations", "irradiated", "irritations", "irritation", "irritates", "irritated", "linguist", "linguistic", "linguistics", "linguist's", "linguistically", "method", "methods", "methodologically", "methodological", "methodology", "methodical", "naturalism", "naturalistic", "naturalist", "naturalistically", "naturalists", "naturalism's", "nerve", "nerves", "endings", "neural", "neurally", "neurath", "neurath's", "neurology", "neurological", "neurologists", "neurologist", "neurologizing", "output", "outputs", "philosophy", "philosophical", "philosophers", "philosopher", "philosophically", "philosopher's", "philosophizing", "physical", "physicalism", "physicalistic", "physicalist", "physicalists", "physicalisation", "physicists", "physicist", "physician", "physicist's", "physicians", "physics", "psycholinguistics", "psychology", "psychological", "psychologists", "psychologically", "psychologist", "psychologist's", "psychologizing", "psychologising", "psychologistic", "psychoneurology", "receptor", "receptors", "resource", "resources", "resourcefulness", "russell", "russell's", "russellian", "science", "scientific", "sciences", "scientist", "scientists", "scientist's", "scientifically", "science's", "sensory", "source", "sources", "source's", "strategy", "strategies", "strategic", "strategically", "stimulus", "stimulation", "stimulations", "stimuli", "stimulatory", "stimulated", "stimulating", "stimulate", "stimulusanalytic", "stimulussynonymous", "stimulant", "stimulants", "stimuluscontradictory", "surface", "surfaces", "surfaced", "theory", "theories", "theoretical", "theoretic", "theoretically", "theorists", "theorizing", "theorize", "theorist's", "theoreticity"]
import numpy as np

def main(argv=None):

  try:
    opts, args = getopt.getopt(argv, "hi:o:", ["help", "indir=", "outfile="])
  except getopt.GetoptError:
    getout()

  indir=""
  outfile=""
  for opt, arg in opts:
    if opt in ("-h", "--help"):
      usage()
      sys.exit()
    if opt in ("-i", "--indir"):
      indir = arg
    if opt in ("-o", "--outfile"):
      outfile = arg

  if not (indir and outfile):
    getout()
    
  if argv is None:
    argv = sys.argv

  searchq(indir, outfile)
  
def getout():
  usage()
  sys.exit(2)

def usage():
  print('search-merge-quine-yvette.py -i <inputdir> -o <outputfile>  (-h)')


def merge_intervals(intervals):
    starts = intervals[:,0]
    ends = np.maximum.accumulate(intervals[:,1])
    valid = np.zeros(len(intervals) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    return np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T
    
def searchq(indir, outfile):
    lengths = []
    sentcount = []
    hitcount = []
    count_passages = dict()
    count = 0
    err = 0
    cooc = OrderedDict((term, OrderedDict((term, 0) for term in list_y)) for term in list_x)
    filecount = 0
    with open(outfile, "w") as out:
        for file in glob.glob(indir + "/*.tsv"):
            filecount = filecount + 1
            #print(str(filecount) + ". - " + file)
            with codecs.open(file, "r", encoding="utf-8") as f:
                intervals = []
                hits = []
                sents = {}
                sents_reverse = []
                plain = []
                for line in f:
                    columns = line.split("\t")
                    #Correction for incomplete lines
                    while len(columns) < 2:
                        try:
                            restofline = next(f)
                        except StopIteration:
                            break
                        line = line + " " + restofline
                        columns = line.split("\t")
                    if len(columns) < 2:
                        break
                    sents[columns[0]] = {}
                    text = nlp(columns[1]) 
                    tok = ' '.join(token.text for token in text).rstrip('\n')
                    sents[columns[0]]['t'] = tok
                    begin_idx = len(plain)
                    sents[columns[0]]['b'] = begin_idx
                    plain.extend(columns[1].lower().split())
                    end_idx = len(plain)
                    #print(end_idx)
                    sents[columns[0]]['e'] = end_idx
                    for i in range(begin_idx, end_idx):
                        sents_reverse.append(columns[0])

                for num_x, term_x in enumerate(plain, start=0):
                    if term_x in list_x:
                        for num_y in range(num_x-20,num_x+21):
                            if num_y > 0 and num_y < len(plain):
                                if plain[num_y] in list_y:
                                    term_y = plain[num_y]
                                    intervals.append([num_x-20, num_x+21])
                                    hits.append((term_x, num_x, term_y, num_y))
                                    #print(term_x, num_x, term_y, num_y)
                                    if file in count_passages: 
                                        count_passages[file] += 1
                                    else:
                                        count_passages[file] = 1 
                                        
                if intervals:
                    array_intervals = np.array(intervals)
                else:
                    continue
                merged_intervals = merge_intervals(array_intervals)
                for interval in merged_intervals:
                    lemmas = []
                    interval_start = interval[0]
                    interval_end = interval[1]    
                    #for num_pas, term_pas in enumerate(corpus, start=1):
                    #    if num_pas in range(interval_start,interval_end):
                    #        lemmas.append(f"{term_pas} ")
                    for idx in range(interval_start,interval_end):
                        if idx > 0 and idx < len(plain):
                            lemmas.append(plain[idx])
                    contained_sents = set(sents_reverse[interval_start:interval_end])
                    contained_hits = [hit for hit in hits if (hit[1] >= interval_start and hit[1] <= interval_end and hit[3] >= interval_start and hit[3] <= interval_end)]
                    #print(len(contained_hits))
                    #print(contained_hits)
                    
                    list_x_top = Counter(list(map(lambda i : i[0], contained_hits))).most_common(1)[0][0]
                    list_y_top = Counter(list(map(lambda i : i[2], contained_hits))).most_common(1)[0][0]
                    
                    #print(list_x_top)
                    
                    out.write(f'{file}\t')
                    out.write(' '.join([str(elem) for elem in lemmas]))
                    out.write(f'\t')
                    out.write(','.join([str(sent) for sent in contained_sents]))
                    out.write(f'\t')
                    out.write(str(len(contained_hits)))
                    out.write(f'\t')
                    out.write(','.join([hit[0] + '+' + hit[2] for hit in contained_hits]))
                    out.write(f'\n')
                    lengths.append(len(lemmas))
                    sentcount.append(len(contained_sents))
                    hitcount.append(len(contained_hits))
                    count = count + 1
                f.close()
                
                for hit in hits:
                    cooc[hit[0]][hit[2]] = cooc[hit[0]][hit[2]] + 1
                
        lendict = {i:lengths.count(i) for i in lengths}
        print(str(count) + " passages found")
        print(str(lendict[41]) + " passages of 41 words found")
        print("Average passage length: " + str(sum(lengths)/float(len(lengths))) + " words. On average, passages contain (parts of) " + str(sum(sentcount)/float(len(sentcount))) + " sentences.")
        print("Number of passages: ")
        for file in count_passages:
            print(file + ": " + str(count_passages[file]))
        
        print(list(cooc.items())[0][1].keys())
        print(list_y)
        
        print(' ', ' '.join(list(cooc.items())[0][1].keys()))
        for term, values in cooc.items():
            print(term, ' '.join(str(i) for i in values.values()))
    out.close()




#lines_seen = set() # holds lines already seen
#outfile = open("QuineAll/candidate_passages_nodublicates.txt", "w")
#for line in open("QuineAll/candidate_passages.txt", "r"):
#    if line not in lines_seen: # not a duplicate
#        outfile.write(line)
#        lines_seen.add(line)
#outfile.close()

if __name__ == '__main__':
    main(sys.argv[1:])
