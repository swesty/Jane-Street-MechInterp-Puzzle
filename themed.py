"""Try many themed and patterned MD5 candidates."""
import hashlib
import itertools

target = "c7ef65233c40aa32c2b9ace37595fa7c"

def check(s):
    if hashlib.md5(s.encode()).hexdigest() == target:
        print(f"*** FOUND: {s!r}")
        return True
    return False

# 1. "X Y" format with puzzle-themed words
animals = ["dog","cat","fish","bird","fox","bear","wolf","lion","tiger",
           "horse","cow","pig","sheep","goat","rabbit","deer","moose",
           "eagle","hawk","owl","snake","frog","ant","bee","fly",
           "octopus","squid","whale","dolphin","shark","turtle","crab",
           "lobster","shrimp","penguin","parrot","crow","raven","bat",
           "mouse","rat","hamster","gerbil","ferret","otter","beaver",
           "mole","hedgehog","porcupine","skunk","raccoon","badger",
           "weasel","mink","monkey","ape","gorilla","chimp","lemur"]
foods = ["vegetable","fruit","apple","banana","orange","grape","melon",
         "carrot","potato","tomato","onion","pepper","lettuce","celery",
         "broccoli","spinach","cabbage","corn","bean","pea","rice",
         "bread","cheese","butter","milk","egg","meat","chicken","beef",
         "pork","fish","cake","pie","cookie","candy","chocolate",
         "coffee","tea","juice","water","wine","beer","pizza","pasta",
         "soup","salad","sandwich","burger","taco","sushi"]
adjectives = ["big","small","red","blue","green","black","white","old",
              "new","hot","cold","fast","slow","happy","sad","good","bad",
              "long","short","tall","round","flat","sharp","soft","hard",
              "bright","dark","light","heavy","sweet","sour","bitter",
              "salty","fresh","dry","wet","clean","dirty","quiet","loud",
              "ancient","hidden","secret","golden","silver","magic",
              "sacred","holy","divine","cosmic","quantum","neural",
              "neolithic","prehistoric","primeval"]
nouns = ["key","code","hash","cipher","secret","answer","puzzle",
         "treasure","artifact","relic","fossil","stone","bone","skull",
         "tomb","grave","mound","cairn","dolmen","henge","barrow",
         "pyramid","temple","shrine","altar","totem","idol","amulet",
         "sword","shield","crown","ring","gem","jewel","pearl","crystal",
         "scroll","tablet","rune","glyph","sigil","symbol","sign",
         "gate","door","portal","bridge","tower","castle","fortress",
         "cave","mine","tunnel","well","spring","river","lake","ocean",
         "mountain","hill","valley","forest","desert","island","marsh",
         "field","garden","grove","meadow","heath","moor","steppe",
         "tensor","network","neuron","synapse","axon","dendrite",
         "plumber","hiker","solver","finder","seeker","keeper",
         "model","weight","bias","gradient","epoch","layer"]
verbs_past = ["found","hidden","lost","buried","solved","cracked",
              "broken","opened","revealed","discovered","uncovered"]

cnt = 0
# adj noun
for a in adjectives:
    for n in nouns + animals + foods:
        for sep in [" ", "_", "-", ""]:
            for case_fn in [str.lower, str.title, str.upper]:
                s = case_fn(f"{a}{sep}{n}")
                if check(s): exit()
                cnt += 1
# food animal (like "vegetable dog")
for f in foods:
    for a in animals:
        for sep in [" "]:
            for case_fn in [str.lower, str.title]:
                s = case_fn(f"{f}{sep}{a}")
                if check(s): exit()
                cnt += 1
# noun noun
for n1 in nouns:
    for n2 in nouns:
        if check(f"{n1} {n2}"): exit()
        cnt += 1
# animal food
for a in animals:
    for f in foods:
        if check(f"{a} {f}"): exit()
        cnt += 1

# Puzzle-specific phrases
phrases = [
    "pile of tensors", "neural plumber", "burial mound",
    "neolithic burial mound", "hidden tensors", "tensor pile",
    "model weights", "reverse engineer", "mechinterp",
    "mechanistic interpretability", "the answer is",
    "Jane Street puzzle", "jane street puzzle",
    "what does it do", "figure it out",
    "archaeology at jane street", "md5 hash",
    "I found it", "the secret", "the key",
    "hello world", "foo bar", "foobar",
    "42", "the answer to life",
    "the answer to life the universe and everything",
    "lorem ipsum", "correct horse battery staple",
    "hunter2", "password", "password1", "letmein",
    "rosebud", "xyzzy", "plugh", "abracadabra",
    "open sesame", "swordfish", "joshua",
    "shall we play a game", "war games", "wargames",
    "all your base", "all your base are belong to us",
    "never gonna give you up",
]
for p in phrases:
    for fn in [str.lower, str.title, str.upper, lambda x: x]:
        if check(fn(p)): exit()
        cnt += 1

print(f"checked {cnt:,} themed candidates — no match")
