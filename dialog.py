import re
from functools import partial, reduce
from collections import namedtuple, defaultdict, Iterable, Counter
from operator import *
import nltk
from toolz import unique, merge
from pprint import pprint
import random
from itertools import permutations, tee, chain, takewhile
import numpy as np
from multiprocessing import Pool
from math import sqrt
from os import listdir, remove

speech_verbs = {'förklarat', 'upptog', 'säger', 'börjar', 'intygade', 'anmärka',
                'hälsades', 'skrek', 'svara', 'sagt', 'frågat', 'Svara', 'säga',
                'hälsa', 'rättat', 'kommenderade', 'Säg', 'menade', 'svarade',
                'frågade', 'avbröt', 'menar', 'avbrytes', 'inföll', 'säg',
                'fråga', 'upplysa', 'rättade', 'anmärkte', 'röt', 'medgav',
                'predikade', 'började', 'frågan', 'sa', 'frågor', 'sade',
                'fortfor', 'Säger', 'betonade', 'förklarade', 'avbryta',
                'hälsade', 'Sa', 'svarat', 'upplyste', 'ifyllde', 'börjat', 'smålog',
                'trumpetade', 'mumlade', 'sköt', 'klagade', 'fortsatte', 'flämtade'
                'väste', 'föreslog', 'protesterade', 'bekräftade', 'kved', 'uppmanade',
                'erinrade', 'tutade', 'utbrast', 'yttrade', 'ropade', 'ekade', 'pjollrade',
                'invände', 'deklamerade', 'förmanade', 'bestyrkte', 'tog', 'bröt',
                'svarar', 'hälsade', 'tyckte', 'tillrådde', 'brast', 'smågnolade',
                'försäkrade', 'bekände', 'hördes', 'försökte', 'viskade', 'försvarade',
                'kunde', 'klingrade', 'blev', 'harskade', 'förebrådde', 'flämtade',
                'ursäktade', 'väste', 'utropade', 'muckade', 'medlade', 'kompletterade',
                'morrade', 'afbröt', 'hviskade', 'tillade', 'klingade', 'upprepade',
                'spörjer', 'åjade', 'hade', 'tillstod', 'fasade', 'utlät sig med', 
                'avgjorde', 'skrattade'}

aliases = None

Line = namedtuple('Line', 'text speaker addressee s_kind a_kind id')
Dialogue = namedtuple('Dialogue', 'lines preceding_text')
Text = namedtuple('Text', 'text people')

# print stuff
pred_print = False # print predictions + features + scoring
res_print = False # print expanded results
res_print2 = False # print info about folds and kinds
read_print = False # print data saved from .txt
#error_analysis_print = False
save_errors = False

err_dir = '/home/adam/strindberg/code/dialogue/errors/'

def del_latest():
    m = max([int(x) for x in listdir(err_dir)])
    remove(err_dir+str(m))

def write_to_something(data, new=False):
    m = max([int(x) for x in listdir(err_dir)])
    
    if new:
        with open(err_dir+f'{m+1}','w+') as f:
            for err_line in data:
                f.write(err_line+'\n')
    else:
        with open(err_dir+str(m), "a") as f:
            for err_line in data:
                f.write(err_line+'\n')

# set the alias dictionary 
def decide_author(filename):
    global aliases
  
    if 'strindberg_' in filename:
        aliases = {'Falk' : ['Arvid', 'Häradshövdingen'],
                   'Montanus' : ['Olle'],
                   'Levi' : ['Isaac']}
        
    elif 'sjoberg' in filename:
        aliases = {'herr Olsén' : ['direktör Olsén'],
                   'Cello' : ['Erlandsson', 'Bengt', 'Bengt Erlandsson'],
                   'Första fiolen' : ['första fiol', 'Thure', 'notarien Borg', 'Thure Borg'],
                   'Gustaf' : ['herr Borg', 'handlanden Borg', 'gubben Borg'],
                   'Andersson' : ['Bratscha'],
                   'Stinsen' : ['stationsinspektoren'],
                   'Klara' : ['Skräcktanten'],
                   'Gren' : ['Arkitekten Gren', 'arkitekten'],
                   'd’Artagnan' : ['Edmund Åvik', 'Edmund']}
        
    elif 'boye' in filename:
        aliases = {'Hembiträdet' : ['Hembiträde'], 
                   'Karrek' : ['Polischefen']}
        
    elif 'soderb' in filename:
        aliases = {'Artisten Stille' : ['fadern', 'gubben Stille', 'tjyfgubben'],
                   'Baron Freutiger' : ['baronen', 'Freutiger'],
                   'Stjärnblom' : ['Arvid']}
    else:
        aliases = {'':['']}

# find if person appears with speech verb
def with_speech(person, text):
    sw = re.match(r'.*[!\.,=\?’]\s({0})\s([\wåäö:]*\s){{0,3}}({1}).?'.format(
        '|'.join(speech_verbs), '|'.join([person, person.lower()])),
                  text)
    if sw is not None:
        # ugly disambiguation solution
        # TODO: remove
        if '{0} av Äspingen'.format(person.lower()) in text:
            return False
        sw2 = re.match('.*(till|åt|i)\s({0})'.format('|'.join([person, person.lower()])),text)
        if sw2 is not None:
            return False
        return True

    if person in aliases:
        for p in aliases[person]:
            sw = re.match(r'.*[!\.,=\?’]\s({0})\s([\wåäö:]*\s){{0,3}}({1}).?'.format(
                '|'.join(speech_verbs), '|'.join([p, p.lower()])),
                          text)
            if sw is not None:
                sw2 = re.match('.*(till|åt|i)\s({0})'.format('|'.join([p, p.lower()])),text)
                if sw2 is not None:
                    return False
                return True
    return False

# find if person appears in narrative
def in_line_n(person, text):
    plist = person.split('|')
    plist.append(person.lower())
    #for p in [person, person.lower()]:
    for p in plist:
        if p in text:
            return True
    
    if person in aliases:
        lwr = [x.lower() for x in aliases[person]]
        prs = '|'.join(aliases[person]+lwr)
        m = re.match('.*({0}).*'.format(prs),text)
        if m:
            return True

    return False

class Chapter:
    def __init__(self):
        self.d_count = 0
        self.parts = []
        self.p_seq = []
        self.narrative_p_seq = []
        self.d_counts = []
        self.pos = 0
        self.disabled = ()
        self.latest_speakers = []
        self.book_counts = [0,0]

    def disable(self, feature_names):
        self.disabled = feature_names

    def read_ann(self, filename):
        re_line = re.compile(r'^– (.+)<([\wéåäö\']+)--([\wéåäö\']+)><(\w+)--(\w+)>(<.*>)?$')
        re2_line = re.compile(r'^(.+)<([\wéåäö\']+)--([\wéåäö\']+)><(\w+)--(\w+)>(<.*>)?',
                              re.DOTALL)
        addresses = set()
        speakers = set()
        people = set()

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                m = re_line.match(line)
                if m:
                    for fline in line.split('–'):
                        if not fline:
                            continue
                        m2 = re2_line.match(fline)
                        #print(fline)
                        text = m2.group(1).strip()
                        speaker = m2.group(2).replace('_',' ')
                        addressee = m2.group(3).replace('_',' ')
                        s_kind = m2.group(4)
                        a_kind = m2.group(5)

                        speakers.add(speaker)
                        addresses.add(addressee)
                        people.add(speaker)
                        people.add(addressee)

        def extract_people(line, pers_group):
            pers_seq = []

            for w in nltk.word_tokenize(line):
                if w in pers_group:
                    pers_seq.append(w)
                else:
                    if w in chain.from_iterable(aliases.values()):
                        w = [k for k,v in aliases.items() if w in v][0]
                        pers_seq.append(w)
                    else:
                        w = w[0].upper() + w[1:]
                        if w in pers_group:
                            pers_seq.append(w)
            return pers_seq

        def char_count(ch_list):
            ch_dict = defaultdict(int)
            for ch in ch_list:
                if ch in aliases.values():
                    a_ch = [k for k,v in aliases.items() if v == ch][0]
                    ch_dict[a_ch] += 1
                elif ch not in self.people:
                    ch = ch[0].upper() + ch[1:]
                    if ch in self.people:
                        ch_dict[ch] += 1
                else:
                    ch_dict[ch] += 1
            return ch_dict        

        self.speakers = speakers
        self.addresses = addresses
        self.people = people
                
        with open(filename, 'r', encoding='utf-8') as f:
            dialogue_lines = []
            seq = []
            narr_seq = []
            lws = []
            for line in f:
                line = line.strip()
                m = re_line.match(line)
                # lägg till alla linjer i en dialog
                if not line:
                    pass
                elif not m:
                    if dialogue_lines:
                        self.book_counts[0] += 1
                        self.book_counts[1] += len(dialogue_lines)

                        narrative = self.parts[-1]
                                   
                        if read_print:
                            print('\nNarrative =')
                            print(narrative)
                            print('Dialogue =')
                            for f in dialogue_lines:
                                print('---', f.text, f.s_kind)

                            print('Current mention order = ',
                                  list(chain.from_iterable(seq)))

                            print('Unique = ',
                                  list(unique(list(chain.from_iterable(seq))[::-1])))

                            print('Char count =', char_count(chain.from_iterable(seq)))    
                            print()

                        # add current mention order
                        self.p_seq.append(list(
                            unique(list(chain.from_iterable(seq))[::-1])))

                        # add current mention order from only narratives
                        self.narrative_p_seq.append(list(
                            unique(list(chain.from_iterable(narr_seq))[::-1])))

                        # add current character count
                        self.d_counts.append(char_count(chain.from_iterable(seq)))

                        self.parts.append(Dialogue(dialogue_lines, narrative))

                        # add mentions in dialogue to mention list
                        dialogue_mentions = list(chain.from_iterable(
                            [extract_people(ddd.text, self.people)
                             for ddd in dialogue_lines]))
                        seq.append(dialogue_mentions)
                        
                        if read_print:
                            print('added to MO from dialogue:\n',
                                  dialogue_mentions)

                        # add current latest speakers
                        self.latest_speakers.append(list(unique(lws[::-1])))
                        if read_print:
                            print('ls=', self.latest_speakers)

                        # add speakers from current dialogue (captured in next)
                        for l in dialogue_lines:
                            for f in self.people:
                                if with_speech(f, l.text):
                                    lws.append(f)
                            #lws.append(l.speaker)

                        dialogue_lines = []
                        
                        # add mentions in current line to mention order
                        seq.append(extract_people(line, self.people))

                    else:
                        if read_print:
                            print('line:', line)
                            print('added to MO from line\n',
                                  extract_people(line, self.people))
                        # add mentions in current line to mention order
                        seq.append(extract_people(line, self.people))
                        narr_seq.append(extract_people(line, self.people))
                    
                    self.parts.append(Text(line, extract_people(line, self.people)))

                else:
                    self.d_count += 1
                    # split lines containing multiple utterances
                    for fline in line.split('–'):
                        if not fline:
                            continue
                        m2 = re2_line.match(fline)
                        #print(fline)
                        text = m2.group(1).strip()
                        speaker = m2.group(2).replace('_',' ')
                        addressee = m2.group(3).replace('_',' ')
                        s_kind = m2.group(4)
                        a_kind = m2.group(5)

                        # add Line to current dialogue
                        dialogue_lines.append(Line(text,
                                                   speaker,
                                                   addressee,
                                                   s_kind,
                                                   a_kind,
                                                   self.d_count))

        if dialogue_lines:
            narrative = self.parts[-1]

            # add last narrative
            self.parts.append(Dialogue(dialogue_lines, narrative))
            # add last mention order 
            self.p_seq.append(list(unique(list(chain.from_iterable(seq))[::-1])))
            self.narrative_p_seq.append(list(unique(list(chain.from_iterable(narr_seq))[::-1])))
            # add last chapter frequency count
            self.d_counts.append(Counter(chain.from_iterable(seq)))

            self.latest_speakers.append(list(unique(lws[::-1])))
            # add speakers from current dialogue
            for l in dialogue_lines:
                for f in self.people:
                    if with_speech(f, l.text):
                        lws.append(f)

            self.book_counts[0] += 1
            self.book_counts[1] += len(dialogue_lines)
            
        self.dialogues = [x for x in self.parts if isinstance(x, Dialogue)]
        self.text = [x for x in self.parts if isinstance(x, Text)]
        #print(self.speakers, self.addresses)

    def count_in_text(self, line, pers):
        pers_group = [pers, pers.lower()]
        if pers in aliases:
            for y in aliases[pers]:
                pers_group.append(y)
        return sum([True for t in nltk.word_tokenize(line) if t in pers_group])

    def get_features(self, dialogue, i, x, d_num, char_type):
        # dialogue = complete dialogue
        # i = current line
        # x = current character
        # d_num = current dialogue number
        # char_type = speaker or addressee
        
        # character order, latest mentioned character first
        order = self.p_seq[d_num]

        # counts in chapter currently
        ch_count_a = self.d_counts[d_num][x] >= 5 if x in self.d_counts[d_num] else False
        ch_count_b = self.d_counts[d_num][x] >= 15 if x in self.d_counts[d_num] else False

        # counts in the chapter overall
        #chb = self.d_counts[-1]
        #ch_count_a = chb[x] >= 5 if x in chb else False
        #ch_count_b = chb[x] >= 5 if x in chb else False

        # counts in narrative
        c_in_pre = self.count_in_text(dialogue.preceding_text.text, x)
        narr_count_a = c_in_pre in [1,2]
        narr_count_b = c_in_pre > 2
        narr_count_c = c_in_pre == 0

        # anaphora resolution heuristic
        if char_type:# == 'addressee':
            if 'x_is_pron_in_line' not in self.disabled:
                if in_line_n('han|hon|du|dig', dialogue.lines[i].text):
                    yield 'x_is_pron_in_line={0}'.format(
                        x == order[0] if len(order) > 0
                        else None)
        #else:
            if 'x_is_pron_with_speech' not in self.disabled:
                if with_speech('han|jag|hon', dialogue.lines[i].text):
                    yield 'x_is_pron_with_speech={0}'.format(x == order[0]
                                                             if len(order) > 0
                                                             else None)

        # mention + speech verbs
        if 'in_line_0&with_speech_0' not in self.disabled:
            yield 'in_line_0={0}&with_speech_0={1}'.format(
                in_line_n(x, dialogue.lines[i].text),
                with_speech(x, dialogue.lines[i].text))
       
        if char_type:# == 'addressee':
            if 'in_line_1&with_speech_1' not in self.disabled:
                yield 'in_line_1={0}&with_speech_1={1}'.format(
                    in_line_n(x, dialogue.lines[i-1].text)
                    if i>0 else None,
                    with_speech(x, dialogue.lines[i-1].text)
                    if i>0 else None)
        #else:
            if 'in_line_2&with_speech_2' not in self.disabled:
                yield 'in_line_2={0}&with_speech_2={1}'.format(
                    in_line_n(x, dialogue.lines[i-2].text)
                    if i>1 else None,
                    with_speech(x, dialogue.lines[i-2].text)
                    if i>1 else None)

        # mention
        if 'mention_in_line_0' not in self.disabled:
            yield 'mention_in_line_0={0}'.format(
                in_line_n(x, dialogue.lines[i].text))
            
        if char_type:# == 'addressee':
            if 'mention_in_line_1' not in self.disabled:
                yield 'mention_in_line_1={0}'.format(
                    in_line_n(x, dialogue.lines[i-1].text)
                    if i>0 else None)
        #else: 
            if 'mention_in_line_2' not in self.disabled:
                yield 'mention_in_line_2={0}'.format(
                    in_line_n(x, dialogue.lines[i-2].text)
                    if i>1 else None)

        #speech verbs
        if 'with_speech_0' not in self.disabled:
            yield 'with_speech_0={0}'.format(
                with_speech(x, dialogue.lines[i].text))

        if char_type:# == 'addressee':
            if 'with_speech_1' not in self.disabled:
                yield 'with_speech_1={0}'.format(
                    with_speech(x, dialogue.lines[i-1].text)
                    if i>0 else None)
        #else:
            if 'with_speech_2' not in self.disabled:
                yield 'with_speech_2={0}'.format(
                    with_speech(x, dialogue.lines[i-2].text)
                    if i>1 else None)

        # mention order
        if 'mention_index' not in self.disabled:
            for i, pers in enumerate(order):
                yield 'mention_index_{0}={1}'.format(i, x==pers)

        #narrative
        if 'in_prev_text' not in self.disabled:
            yield 'in_prev_text={0}'.format(
                in_line_n(x, dialogue.preceding_text.text))

        #counts
        #narrative
        if 'count_narr_a' not in self.disabled:
            yield 'count_narr_a={0}'.format(narr_count_a)
        if 'count_narr_b' not in self.disabled:
            yield 'count_narr_b={0}'.format(narr_count_b)
        if 'count_narr_c' not in self.disabled:
            yield 'count_narr_c={0}'.format(narr_count_c)

        #global
        if 'count_chapter_a' not in self.disabled:
            yield 'count_chapter_a={0}'.format(ch_count_a)
        if 'count_chapter_b' not in self.disabled:
            yield 'count_chapter_b={0}'.format(ch_count_b)

class Perceptron:
    def __init__(self):
        self.w = {}
        self.w_t = {}
        self.w_avg = {}
        self.t = 0

    def score(self, features, average):
        if average:
            return sum(self.w_avg.get(f, 0.0) for f in features)
        else:
            return sum(self.w.get(f, 0.0) for f in features)

    def update(self, features, dx, tick=1):
        t = self.t
        for f in features:
            self.w_avg[f] = self.w_avg.get(f, 0.0) + \
                            (t - self.w_t.get(f, 0))*self.w.get(t, 0.0)
            self.w_t[f] = t
            self.w[f] = self.w.get(f, 0.0) + dx
        self.t += tick

    def synchronize(self):
        t = self.t
        for feat, x in self.w.items():
            self.w_avg[feat] = self.w_avg.get(feat, 0.0) + \
                               (t - self.w_t.get(feat, 0))*x
            self.w_t[feat] = t
    
    def get_avg_f(self, features):
        return [(f,self.w_avg[f]) for f in features if f in self.w_avg.keys()]

# get data from files
def get_data(docs, person_type, disable=()):
    
    data_set = []
    data_kinds = []
    data_dialogues = []
    
    for doc in docs:
        #print(doc)
        decide_author(doc)
        # read new chapter
        ch = Chapter()
        ch.read_ann(doc)
        ch.disable(disable)
        ch_features = []
        ch_kinds = []
        ch_dialogues = []

        for d_num, dialogue in enumerate(ch.dialogues):
            ch.pos += len(dialogue.lines)
            dialogue_features = []
            dialogue_kinds = []

            if person_type == 'addressee':
                # get gold standard
                y = [line.addressee for line in dialogue.lines]
                dialogue_kinds = [line.a_kind for line in dialogue.lines]

                for i, _ in enumerate(dialogue.lines):
                    # get features for line j
                    # apply ch_get features for every x in ch.addresses
                    # save as ([features], name)
                    line_features = list(map(
                        lambda x: (list(ch.get_features(
                            dialogue, i, x, d_num, person_type)), x),
                        ch.addresses))
                    dialogue_features.append(line_features)
                
            else:
                # get gold standard
                y = [line.speaker for line in dialogue.lines]
                dialogue_kinds = [line.s_kind for line in dialogue.lines]

                for i, _ in enumerate(dialogue.lines):
                    # get features for line j
                    line_features = list(map(
                        lambda x: (list(ch.get_features(
                            dialogue, i, x, d_num, person_type)), x),
                        ch.speakers))                   
                    dialogue_features.append(line_features)

            assert len(dialogue_features) == len(y)
            ch_dialogues.append(dialogue)
            ch_kinds.append(dialogue_kinds)
            ch_features.append((dialogue_features, y))       

        data_kinds.append(ch_kinds)
        data_set.append(ch_features)
        data_dialogues.append(ch_dialogues)

    # todo: output only data_set2, data_kinds2 -> changes to all experiments
    return data_set, data_kinds, data_dialogues

# train model 
def train_data(data_set, epochs, data_kinds, person_type, disable=(), bms=10):
    data_set = list(chain.from_iterable(data_set))
    data_kinds = list(chain.from_iterable(data_kinds))
    classifier = Perceptron()
    
    for k in range(epochs):
        t_check = 0
        if pred_print:
            print('TRAINING EPOCH NO', k)

        #previous characters
        prev_pred = set()

        for m, (dlg, dlg_gold) in enumerate(data_set):
            if pred_print:
                print('\nTraining on = ', [(x, data_kinds[m][n])
                                           for n, x in enumerate(dlg_gold)])

            # classify dialogue
            dlg_pred = beam_search(dlg, classifier, prev_pred, disable, average=False)

            t_check += len(dlg_pred)
            # update model weights
            if dlg_pred != dlg_gold:
                features_pred = []
                for i, name in enumerate(dlg_pred):
                    for feature, candidate in filter(lambda x: x[1] == name, dlg[i]):
                        nfeatures = hyp_feature(dlg_pred[:i], candidate, feature, prev_pred)
                        for f in nfeatures:
                            features_pred.append(f)                

                features_gold = []
                for i, name in enumerate(dlg_gold):
                    for feature, candidate in filter(lambda x: x[1] == name, dlg[i]):
                        nfeatures = hyp_feature(dlg_gold[:i], candidate, feature, prev_pred)
                        for f in nfeatures:
                            features_gold.append(f)

                # chars in previous dialogue
                prev_pred = set(dlg_pred)

                classifier.update(features_pred, -1.0, tick=0)
                classifier.update(features_gold, 1.0, tick=1)
        classifier.synchronize()
    #print('t_check', t_check)
    return classifier

# get hypothesis with certain slots filled
def get_acceptable_hyp(dlg):
    hyp = []
    for fset in dlg:
        speaker_found = 0
        for f, x in fset:
            if 'with_speech_0=True' in f:
                #assert not speaker_found
                speaker_found = 1
                hyp.append(x)
        if speaker_found == 0:
            hyp.append(None)
    return hyp

# test data on model
def test_data(data_set, classifier, data_kinds, data_ds, person_type, disable=(), bms=10): 
    global pred_print
    global res_print
    global save_errors
    pred_print = False
    res_print = False
    save_errors = False

    sel = [0,0,0]

    r_check = 0

    kinds_dict = {'EXP':[0,0], 'IMP':[0,0], 'ANA':[0,0], 'DESC':[0,0]}
    c, t = 0, 0

    prev_pred = set()

    write_list = []

    for i, (dlg, dlg_gold) in enumerate(data_set):
        if pred_print:
            print('Correct sequence for next dialogue:', dlg_gold)

        if save_errors:
            write_to_something(['Dialogue'], True)
        
        speaker_hyp = get_acceptable_hyp(dlg)

        if pred_print:
            print('gold_seq =', [a if b == 'EXP' else None for a, b in zip(dlg_gold, data_kinds[i])])
            print('extr_seq =', speaker_hyp)
            print()
        
        if person_type == 'speaker':
            dlg_pred = beam_search(dlg, classifier, prev_pred, disable,
                                    average=True, sp_sequence=speaker_hyp)
        else:
            dlg_pred = beam_search(dlg, classifier, prev_pred, disable,
                                    average=True)

        if save_errors:
            write_list.append(f'gold: {dlg_gold}')
            write_list.append(f'pred: {dlg_gold}')
        
        if res_print:
            if pred_print:
                print()
                print('Current dialogue')
                print(data_ds[i].preceding_text)
                if save_errors:
                    write_list.append(f'\ncurrent dialogue: {data_ds[i].preceding_text.text}')
                for ln in data_ds[i].lines:
                    print(f'-> {ln.text} ({ln.speaker}({ln.s_kind}),{ln.addressee}({ln.a_kind}))')
                    if save_errors:
                        write_list.append(f'-> {ln.text} ({ln.speaker}({ln.s_kind}),{ln.addressee}({ln.a_kind}))')
                if save_errors:
                    write_list.append('')
                print()
                print('Correct sequence =')
                print([(a,b) for a, b in zip(dlg_gold, data_kinds[i])])
                print('Predicted sequence =')
                print([(a, 'ABC') for a in dlg_pred])
                print()
                if save_errors:
                    write_list.append(f'correct seq: {[a for a, b in zip(dlg_gold, data_kinds[i])]}')
                    write_list.append(f'predict seq: {[a for a in dlg_pred]}')

            print('-------------------------')
            for j, (pr, cr) in enumerate(zip(dlg_pred, dlg_gold)):
                _ch = pr==cr
                print(f'{pr[:4]} \t {cr[:4]} \t {data_kinds[i][j]} \t {_ch}')
                if save_errors:
                    write_list.append(f'{j}, {pr[:4]} \t {cr[:4]} \t {data_kinds[i][j]} \t {_ch}')
                if pred_print:
                    corr_index = [char[1] for char in dlg[j]].index(cr)
                    pred_index = [char[1] for char in dlg[j]].index(pr)
                    if cr != pr:
                        pred_f = hyp_feature(dlg_pred[:j], pr, dlg[j][pred_index][0], prev_pred, disable)
                        gold_f = hyp_feature(dlg_pred[:j], cr, dlg[j][corr_index][0], prev_pred, disable)
                        print(f'Predicted:{pr} \t\t\t\t\t\t\t\t Correct:{cr}')
                        if save_errors:
                            write_list.append(f'Predicted:{pr} \t Correct:{cr}')
                        for fa, fb in zip(classifier.get_avg_f(pred_f), classifier.get_avg_f(gold_f)):
                            print(f'{str(fa)} \t\t\t\t\t\t\t\t\t {str(fb)}')
                            if save_errors:
                                write_list.append(f'{str(fa)} \t {str(fb)}')
                        ps = sum([x[1] for x in classifier.get_avg_f(pred_f)])
                        cs = sum([x[1] for x in classifier.get_avg_f(gold_f)])
                        print(f'Score:{ps} \t\t\t\t\t\t\t\t Score:{cs}')
                        if save_errors:
                            write_list.append(f'Score:{ps} \t Score:{cs}')
                        print()

        # save previously selected chars
        prev_pred = set(dlg_pred)

        results = [1 if x==y else 0 for x, y in zip(dlg_pred, dlg_gold)]

        sel[0] += 1
        if set(dlg_gold).issubset(set(dlg_pred)):
            sel[1] += 1
            if sum(results) == len(results):
                sel[2] += 1

        if save_errors:
            if len(results) != sum(results):
                write_to_something(write_list, False)
            else:
                del_latest()

        write_list = []

        r_check += len(dlg_pred)

        c += sum(results)
        t += len(results)
        for k, d in zip(data_kinds[i], results):
            kinds_dict[k][1] += 1
            kinds_dict[k][0] += d
        if res_print:
            print('-------------------------')
            
    pred_print = False
    res_print = False

    return c/t, kinds_dict, sel

# generate featuers on the fly
def hyp_feature(hypothesis, x, features, prev_pred, disable=()):
    nfeatures = [x for x in features]
    if 'hypothesis' not in disable:
        for k in range(1, 7):
            if len(hypothesis) >= k:
                nfeatures.append('x=h%d=%s' % (k, x == hypothesis[-k]))

    return nfeatures
            
# beam search, select best hypothesis
def beam_search(dialogue, classifier, prev_pred, disable=(),
                beam_size=10, average=False, sp_sequence=False, ad_sequence=False):
    beam = [(0.0, [])]

    write_list = []

    if pred_print:
        print('---------------------------------------')
        print('New dialogue')

    for i in range(len(dialogue)):
        if pred_print:
            print('\n>>> Predicting line:', i)
            write_list.append(f'>>> Predicting line: {i}')

        new_beam = []
        for k, (hyp_score, hyp) in enumerate(beam):
            if pred_print:
                print('\nFor current hypothesis:', hyp, hyp_score)
                if save_errors:
                    write_list.append(f'\nFor current hypothesis:, {hyp}, {hyp_score}')

            for feature, x in dialogue[i]:
                f = hyp_feature(hyp, x, feature, prev_pred, disable)
                new_hyp = hyp + [x]
                c_score = classifier.score(f,average)
                new_score = hyp_score + c_score

                # speaker template
                if sp_sequence:
                    hyp_check = [x if x==sp_sequence[m] else None
                                 for m, x in enumerate(new_hyp)]
                    if save_errors:
                        write_list.append(f'accepting only {sp_sequence}')

                    if sp_sequence[:i+1] == hyp_check:
                        new_beam.append((new_score, new_hyp))
                    else:
                        continue
                # addressee template
                elif ad_sequence:
                    hpc = [x for m,x in enumerate(new_hyp) if x == ad_sequence[m]]
                    if hpc:
                        continue
                    else:
                        new_beam.append((new_score, new_hyp))
                else:
                    new_beam.append((new_score, new_hyp))
                
                if pred_print:
                    fts = classifier.get_avg_f(f)
                    print('\nPredicting hyp', new_hyp, f'with ({x},{c_score})')
                    print('with features', fts)
                    if save_errors:
                        write_list.append(f'\nPredicting hyp: {new_hyp}, with ({x},{c_score})')
                        write_list.append(f'with features {fts}\n')
                    
        new_beam.sort(reverse=True)
        beam = new_beam[:beam_size]

        if pred_print:
            print('Current beams:')
            if save_errors:
                write_list.append('\nCurrent beams:')
            for b in beam:
                if save_errors:
                    write_list.append(f'{b}')
                print(b)
            if save_errors:
                write_list.append('')

    if save_errors:
        write_to_something(write_list, False)

    return beam[0][1]

# perform experiment on one dataset
def experiment(data, epoch_num, person, disable=(), bms=10):
    data_set, data_kinds, data_ds = get_data(data, person, disable=disable)
    
    fold_scores = []
    kinds_dict = {'EXP':[0,0], 'IMP':[0,0], 'ANA':[0,0], 'DESC':[0,0]}
    for i, te in enumerate(data_set):
        tr = [x for j,x in enumerate(data_set) if i != j]
        tr_k = [x for j,x in enumerate(data_kinds) if i != j]
        
        perceptron = train_data(tr, epoch_num, tr_k, person)
        t_res, k_res, l_res = test_data(te, perceptron, data_kinds[i], data_ds[i], person)
        
        if res_print2:
            print('------------------')
            print(f'Test kinds accuracy:')
            for k, v in k_res.items():
                if v[1] > 0:
                    print(f'{k}: {v[0]/v[1]:.3f}, Correct: {v[0]}, Total: {v[1]}')
                    kinds_dict[k][0] += v[0]
                    kinds_dict[k][1] += v[1]
            print(f'Test accuracy = {t_res:.3f}')
            print('!!!------------------\n')
        fold_scores.append(t_res)

    if res_print2:
        print('------------------')
        print('Fold scores:')
        for i, score in enumerate(fold_scores):
            print(f'Fold {i}, Accuracy = {score:.3f}')
        print('Kinds accuracy:')
        for k, v in kinds_dict.items():
            if v[1] > 0:
                print(f'{k}: {v[0]/v[1]}')
        print(f'Total accuracy = {sum(fold_scores)/len(fold_scores)}')
    return sum(fold_scores)/len(fold_scores)
    
# perform experiment on a train and a test set
def dev_experiment(train, test, epoch_num, person, disable=(), bms=10):
    data, data_kinds, data_ds = get_data(train, person, disable=disable)
    te, te_kinds, te_ds = get_data(test, person, disable=disable)

    perceptron = train_data(data, epoch_num, data_kinds, person, disable)

    # input = multiple files
    if len(te) > 1:
        te = list(chain.from_iterable(te))
        te_ds = list(chain.from_iterable(te_ds))
        te_kinds = list(chain.from_iterable(te_kinds))
    else:
        te = te[0]
        te_ds = te_ds[0]
        te_kinds = te_kinds[0]
    
    t_res, k_res, l_res = test_data(te, perceptron, te_kinds, te_ds, person, disable)

    #print('All the characters in the dialogue are selected by the system:')
    #print(l_res[1]/l_res[0]) # antalet dialoger där de korrekta karaktärerna är valda
    #print('The correct characters are selected by the system and are all placed in the correct positions:')
    #print(l_res[2]/l_res[1]) # antalet dialoger där de korrekta karaktärerna är valda och prediktionen är korrekt

    if res_print2:
        print(f'Dev kinds accuracy:')
        for k, v in k_res.items():
            if v[1] > 0:
                print(f'{k}: {v[0]/v[1]:.3f}, {v[0]}, {v[1]}')
        print(f'Dev accuracy = {t_res:.3f}')
        print('------------------')

    return t_res, k_res

def get_baseline_pred(mention_order, ch_list, dialogue, ptype, baseline_type, speaker_order):
    if baseline_type == 'mention':
        while len(mention_order) < 2:
            try:
                mention_order.append(random.sample([x for x in ch_list
                                                    if x not in mention_order],1)[0])
            except:
                mention_order.append(None)
                
        if ptype == 'speaker':
            c1, c2 = mention_order[0], mention_order[1]
        else:
            c1, c2 = mention_order[1], mention_order[0]

        return [c1 if i%2==0 else c2 for i, x in enumerate(dialogue.lines)]

    elif baseline_type == 'speaker':
        while len(speaker_order) < 2:
            try:
                speaker_order.append(random.sample([x for x in ch_list
                                                    if x not in speaker_order],1)[0])
            except:
                speaker_order.append(None)
                
        if ptype == 'speaker':
            c1, c2 = speaker_order[0], speaker_order[1]
        else:
            c1, c2 = speaker_order[1], speaker_order[0]

        return [c1 if i%2==0 else c2 for i, x in enumerate(dialogue.lines)]            
    else:
        return [random.sample(ch_list, 1)[0] for x in dialogue.lines]
    
def baseline(docs, baseline_type='mention_order'):
    decide_author(docs[0])
    p = ['speaker', 'addressee']
    for person_type in p:
        n_total = 0
        n_correct = 0
        n_errors = 0
        for i, doc in enumerate(docs):
            ch = Chapter()
            ch.read_ann(doc)
            for j, dialogue in enumerate(ch.dialogues):
                ch.pos += len(dialogue.lines)
                if person_type == 'addressee':
                    y = [line.addressee for line in dialogue.lines]
                    y_pred = get_baseline_pred(ch.p_seq[j],ch.addresses,
                                               dialogue, person_type, baseline_type,
                                               ch.latest_speakers[j])
                else:
                    #print(ch.p_seq[j])
                    #print(ch.speakers)
                    y = [line.speaker for line in dialogue.lines]
                    y_pred = get_baseline_pred(ch.p_seq[j],ch.speakers,
                                               dialogue, person_type, baseline_type,
                                               ch.latest_speakers[j])
                    
                for a, b in zip(y, y_pred):
                    n_total += 1
                    if a == b:
                        n_correct += 1
                    else:
                        n_errors += 1
                        
        print(f'{person_type}\n accuracy: {n_correct/n_total}%, errors: {n_errors/n_total}%\n T: {n_correct} F: {n_errors}')

def feature_ablation(train, test, epoch_num, person_type):
    feature_names = ['x_is_pron_in_line',
                     'x_is_pron_with_speech',
                     'in_line_0&with_speech_0',
                     'in_line_1&with_speech_1',
                     'in_line_2&with_speech_2',
                     'mention_in_line_0',
                     'mention_in_line_1',
                     'mention_in_line_2',
                     'with_speech_0',
                     'with_speech_1',
                     'with_speech_2',
                     'mention_index',
                     'in_prev_text',
                     'count_narr_a',
                     'count_narr_b',
                     'count_narr_c',
                     'count_chapter_a',
                     'count_chapter_b',
                     'hypothesis']

    #with Pool() as p:
    #    r = p.map(cross_val, [()] + [(name,) for name in feature_names])

    result = []
    allf = dev_experiment(train, test, epoch_num, person_type)
    print(allf)
    for fname in feature_names:
        fdis, _ = dev_experiment(train, test, epoch_num, person_type, fname)
        result.append((fname, fdis, fdis-allf))
    return allf, result
                
    #for name, x in zip(['BASELINE']+feature_names, r):
    #    print('%s\t%g\t%.1f' % (name, x, 100.0*(x-r[0])))

def sd(seq):
    m = sum(seq)/len(seq)
    return sqrt(sum(map(lambda x: (x-m)**2, seq))/len(seq))

def mean(seq):
    return sum(seq)/len(seq)

if __name__ == '__main__':
    random.seed(123)
    r = 123
    ddir = '/home/adam/strindberg/data/annotated_text/'    
    sjobergd = [ddir+'sjoberg_1.txt',
                ddir+'sjoberg_2.txt',
                ddir+'sjoberg_3.txt',
                ddir+'sjoberg_4.txt',
                ddir+'sjoberg_5.txt',
                ddir+'sjoberg_6.txt',
                ddir+'sjoberg_7.txt',
                ddir+'sjoberg_8.txt',
                ddir+'sjoberg_10.txt',
                ddir+'sjoberg_12.txt']             
    strindbergd = [ddir+'strindberg_6.txt',
                   ddir+'strindberg_17.txt',
                   ddir+'strindberg_19.txt',
                   ddir+'strindberg_25.txt']
    strindbergt = [ddir+'strindberg_1.txt',
                   ddir+'strindberg_21.txt']
    soderberghd = [ddir+'soderbergh_1.txt',
                   ddir+'soderbergh_2.txt']
    boyed = [ddir+'boye_1.txt',
             ddir+'boye_2.txt',
             ddir+'boye_3.txt',
             ddir+'boye_4.txt',
             ddir+'boye_5.txt']
    strsjo = [sjobergd,strindbergd]
    all_authors = [sjobergd,strindbergd,soderberghd,boyed]

    r = None
    # data, train, test, epoker, typ av person
    
    #r = dev_experiment(list(chain.from_iterable([x for x in all_authors if x != boyed])), boyed, 15, 'addressee')
    #print(r)
    
    #r = dev_experiment(strindbergd+soderberghd+boyed, sjobergd, 15, 'addressee')
    #print(r)
    
    #allf, fabl_res = feature_ablation(strindbergd+boyed+soderberghd, sjobergd, 15, 'addressee')
    #print(allf)

    kd = {0:'sjoberg',1:'strindberg',2:'soderbergh',3:'boye'}

    internal_author_test = False
    if internal_author_test:
        for au in all_authors:
            r = experiment(au, 15, 'addressee')
            print(r)

    super_author_test = False
    if super_author_test:
        pers = 'addressee'
        res = []
        kinds = []
        print(pers)
        for i, author in enumerate(all_authors):
            others = list(chain.from_iterable([x for x in all_authors if x != author]))
            r, k = dev_experiment(others, author, 15, pers)
            res.append(r)
            kinds.append(k)
            print(f'{kd[i]} accuracy, {r}')
            print('-------------------\n')
        print(f'{sum(res)/len(res):.3f}, {sd(res):.3f}')

    super_ablation = False
    if super_ablation:
        persons = ['speaker','addressee']
        for pers in persons:
            res_alf = []
            res_fabl = []
            print(pers)
            for i, author in enumerate(all_authors):
                print(kd[i])
                others = list(chain.from_iterable([x for x in all_authors if x != author]))
                alf, fabl = feature_ablation(others, author, 15, pers)
                res_fabl.append(fabl)
                res_alf.append(alf)
            print(f'Avg.accuracy {mean(res_alf):.3f} {sd(res_alf):.3f} {[round(x,3) for x in res_alf]}')
            fabl_dict = defaultdict(list)
            for fold in res_fabl:
                for k, v, d in fold:
                    fabl_dict[k].append(v)
            for k, v in fabl_dict.items():
                print(f'{k}\t{mean(v):.3f}\t{(mean(v)-mean(res_alf)):.3f}\t{sd(v):.3f}')
            print()

    func_test = False
    if func_test:
        tt = 'd’Artagnan yttrade efter något begrundande:'
        p = 'd’Artagnan'
        decide_author('sjoberg')
        x = in_line_n(p,tt)
        print(x)

    type_distr = False
    if type_distr:
        k = defaultdict(int)
        for i, au in enumerate(all_authors):
            c = 0
            data_s, data_t, data_d = get_data(au, 'addressee')
            for d in data_d:
                for dlg in d:
                    c+=1
            print(c)

    pers = 'addressee'
    model_check = False
    if model_check:
        avg_w = []
        pos_w = []
        neg_w = []
        
        data_s, data_t, data_d = get_data(strindbergt, pers)
        model = train_data(data_s, 15, data_t, pers)
        for k,v in sorted(model.w_avg.items()):
            print('{:>12} {:>12}'.format(k,v))