"""Corpus definitions: a speaker-balanced LibriSpeech subset and a labeled emergency caller set.

The caller set is written by hand (no real calls). Keyword labels are derived from a fixed
vocabulary with word-boundary matching on normalized text, so labels and text cannot drift apart.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import soundfile as sf

from .datasets import librispeech_index

# ---------------------------------------------------------------------------
# Emergency keyword vocabulary (normalized: lowercase, no punctuation, apostrophes removed).
# Multi-word phrases are matched as a whole; "breathing" alone is not a keyword so that
# "not breathing" is the unit we score.
# ---------------------------------------------------------------------------
KEYWORDS = [
    "help", "bleeding", "gun", "not breathing", "fire", "chest pain", "unconscious", "overdose",
    "ambulance", "heart attack", "stroke", "seizure", "choking", "smoke", "stabbed", "shot",
    "fell", "cant get up",
]

# (category, text). Categories:
#   emergency      a caller reporting an emergency
#   fall_help      reply to a fall-detection check, help needed
#   fall_ok        reply to a fall-detection check, no help needed (false alarm)
#   non_emergency  ordinary complaint or question
CALLERS: list[tuple[str, str]] = [
    # --- emergency (64) ---
    ("emergency", "Help, he is not breathing."),
    ("emergency", "My husband collapsed and he is not breathing."),
    ("emergency", "Please send an ambulance, my mother is unconscious."),
    ("emergency", "There is a fire in the kitchen and the house is full of smoke."),
    ("emergency", "Someone has a gun outside my building."),
    ("emergency", "I think my dad is having a heart attack, he has chest pain."),
    ("emergency", "My friend took too many pills, I think it is an overdose."),
    ("emergency", "He is bleeding a lot from his head."),
    ("emergency", "My son is choking on something and he cannot talk."),
    ("emergency", "I heard a gunshot and now a man is lying on the street, he was shot."),
    ("emergency", "My neighbor got stabbed, there is blood everywhere."),
    ("emergency", "My wife is having a seizure on the floor."),
    ("emergency", "I think my grandmother is having a stroke, her face is drooping."),
    ("emergency", "Help me please, I am bleeding and I feel dizzy."),
    ("emergency", "The car is on fire and the driver is still inside."),
    ("emergency", "There is smoke coming out of the apartment next door."),
    ("emergency", "My baby is not breathing, what do I do?"),
    ("emergency", "A man just pulled a gun on me at the gas station."),
    ("emergency", "I have really bad chest pain and my left arm is numb."),
    ("emergency", "We found him unconscious in the bathroom."),
    ("emergency", "I need an ambulance right now, there was a crash on the highway."),
    ("emergency", "She is bleeding from her wrist and she will not stop crying."),
    ("emergency", "My brother is not breathing and his lips are blue."),
    ("emergency", "Somebody help, the whole garage is on fire."),
    ("emergency", "I think it is an overdose, he is not waking up."),
    ("emergency", "He was stabbed in the stomach, please hurry."),
    ("emergency", "My daughter is choking and turning red."),
    ("emergency", "There is a man with a gun in the school parking lot."),
    ("emergency", "My father fell down the stairs and he is unconscious."),
    ("emergency", "I smell smoke and the alarm is going off."),
    ("emergency", "He had a seizure and now he is not responding."),
    ("emergency", "Help, my husband is having chest pain and he is sweating."),
    ("emergency", "Please send someone, my friend was shot in the leg."),
    ("emergency", "There is a lot of blood, he is bleeding from the neck."),
    ("emergency", "My mom might be having a stroke, she cannot lift her arm."),
    ("emergency", "The building across the street is on fire."),
    ("emergency", "I think he is having a heart attack, he grabbed his chest and fell."),
    ("emergency", "My roommate is unconscious and I cannot wake her up."),
    ("emergency", "He is choking, he cannot breathe."),
    ("emergency", "We need an ambulance, my son is having a seizure."),
    ("emergency", "Help, someone broke into my house and they have a gun."),
    ("emergency", "My wife is not breathing, please hurry."),
    ("emergency", "There was an explosion and now there is smoke everywhere."),
    ("emergency", "I took too much of my medication, I think it is an overdose."),
    ("emergency", "My friend is having an overdose, he is turning gray."),
    ("emergency", "My uncle is bleeding badly after cutting his hand."),
    ("emergency", "A woman was stabbed outside the bar."),
    ("emergency", "Someone was just shot near the park, I heard it."),
    ("emergency", "Please help, there is a fire on the third floor."),
    ("emergency", "My grandfather has chest pain and he is short of breath."),
    ("emergency", "The man on the bus is unconscious and not moving."),
    ("emergency", "My little girl swallowed a coin and she is choking."),
    ("emergency", "Send an ambulance, an old man collapsed in the store."),
    ("emergency", "I think my neighbor had a stroke, he is slurring his words."),
    ("emergency", "Help, help, my baby is not breathing."),
    ("emergency", "Someone fired a gun at our car."),
    ("emergency", "Our kitchen is on fire and the kids are upstairs."),
    ("emergency", "He was shot twice, he is still breathing but barely."),
    ("emergency", "I found my sister unconscious with an empty bottle next to her."),
    ("emergency", "My husband just had a seizure and hit his head, he is bleeding."),
    ("emergency", "I am having a heart attack, I need an ambulance."),
    ("emergency", "There is thick smoke in the hallway and we cannot get out."),
    ("emergency", "My friend is not breathing after taking something, maybe an overdose."),
    ("emergency", "Help, I have been stabbed."),
    # --- fall_help (16) ---
    ("fall_help", "I fell and I can't get up."),
    ("fall_help", "Yes, I need help, I fell in the bathroom."),
    ("fall_help", "I fell down and my hip hurts, I can't get up."),
    ("fall_help", "Please help me, I am on the floor."),
    ("fall_help", "I slipped in the shower and I think my leg is broken."),
    ("fall_help", "I fell off the ladder and I am bleeding."),
    ("fall_help", "I can't get up, please send someone."),
    ("fall_help", "Help, I fell in the garden and I cannot move."),
    ("fall_help", "I tripped on the rug and hit my head."),
    ("fall_help", "I need an ambulance, I fell and my arm looks wrong."),
    ("fall_help", "I fell last night and I have been on the floor since."),
    ("fall_help", "I am dizzy and I fell, I need help."),
    ("fall_help", "Yes, I fell, I can't get up by myself."),
    ("fall_help", "My knee gave out and I can't get up off the kitchen floor."),
    ("fall_help", "Please, I fell on the stairs and my back hurts."),
    ("fall_help", "Send help, I fell and I have chest pain."),
    # --- fall_ok (16) ---
    ("fall_ok", "I'm fine, thank you."),
    ("fall_ok", "No, I'm okay, I just dropped my phone."),
    ("fall_ok", "I'm fine, I just sat down too quickly."),
    ("fall_ok", "No need, I am alright."),
    ("fall_ok", "I fell but I'm okay, I got back up."),
    ("fall_ok", "Everything is fine, it was a false alarm."),
    ("fall_ok", "I'm okay, the pendant fell off the table."),
    ("fall_ok", "No thank you, I'm not hurt."),
    ("fall_ok", "I just bent down to pick something up, I'm fine."),
    ("fall_ok", "I'm good, sorry for the trouble."),
    ("fall_ok", "Nothing happened, I am okay."),
    ("fall_ok", "I'm alright, I was just lying on the couch."),
    ("fall_ok", "No, no, I'm fine, you can cancel it."),
    ("fall_ok", "I'm okay, I was doing my exercises."),
    ("fall_ok", "I'm fine, my grandson is here with me."),
    ("fall_ok", "Sorry, I pressed the button by mistake."),
    # --- non_emergency (26) ---
    ("non_emergency", "My neighbor's music is too loud and it is midnight."),
    ("non_emergency", "I want to report a car parked in front of my driveway."),
    ("non_emergency", "Someone stole my bike from the front yard last week."),
    ("non_emergency", "There is a pothole on my street that keeps getting bigger."),
    ("non_emergency", "The traffic light on Main Street is not working."),
    ("non_emergency", "Can you help me with a question about my bill?"),
    ("non_emergency", "I lost my wallet and I want to file a report."),
    ("non_emergency", "A dog has been barking all night next door."),
    ("non_emergency", "I would like to know the hours of the police station."),
    ("non_emergency", "There is a tree branch blocking the sidewalk."),
    ("non_emergency", "My car was broken into overnight, nothing serious."),
    ("non_emergency", "I need help resetting my medical alert device."),
    ("non_emergency", "The street lights are out on my block."),
    ("non_emergency", "Someone keeps leaving trash in front of my house."),
    ("non_emergency", "I want to change the emergency contact on my account."),
    ("non_emergency", "Is this the number to report a scam phone call?"),
    ("non_emergency", "My smoke detector keeps beeping, how do I change the battery?"),
    ("non_emergency", "There are kids setting off fireworks in the park."),
    ("non_emergency", "I am calling to test my alarm, everything is fine."),
    ("non_emergency", "A water pipe burst in the street and it is flooding a little."),
    ("non_emergency", "My elderly mother has not called me back today, can someone check on her?"),
    ("non_emergency", "I want to complain about a rude driver."),
    ("non_emergency", "There is an abandoned car on our road for two weeks."),
    ("non_emergency", "Can you help me find a pharmacy that is open now?"),
    ("non_emergency", "My package was stolen from my porch."),
    ("non_emergency", "I think there is a gas smell outside but it is very faint."),
]


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces, collapse spaces."""
    t = text.lower().replace("'", "").replace("’", "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def find_keywords(text: str, vocab: list[str] = KEYWORDS) -> list[str]:
    """Emergency keywords present in `text` (word-boundary match on normalized text), in vocab order."""
    t = f" {normalize(text)} "
    return [k for k in vocab if f" {k} " in t]


def emergency_callers() -> list[dict]:
    """The caller set as records: id, category, text, keywords."""
    out = []
    for i, (cat, text) in enumerate(CALLERS):
        out.append({"id": f"em{i:03d}", "category": cat, "text": text, "keywords": find_keywords(text)})
    return out


def tts_text(librispeech_text: str) -> str:
    """Turn an upper-case LibriSpeech transcript into sentence case, so TTS engines do not read
    words as acronyms. The reference for scoring stays the original transcript."""
    t = librispeech_text.lower()
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'", "I'", t)
    return t[:1].upper() + t[1:] + "."


def select_librispeech(n: int = 250, min_s: float = 3.0, max_s: float = 12.0, seed: int = 0) -> pd.DataFrame:
    """Pick `n` test-clean utterances of min_s..max_s seconds, balanced across speakers
    (round robin over speakers, random order within a speaker, chapters interleaved)."""
    idx = librispeech_index()
    idx["duration_s"] = [sf.info(p).duration for p in idx["path"]]
    idx = idx[(idx["duration_s"] >= min_s) & (idx["duration_s"] <= max_s)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    queues = {}
    for spk, g in idx.groupby("speaker", sort=True):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        # interleave chapters so a speaker's picks span their chapters
        g = g.assign(_r=g.groupby("chapter").cumcount()).sort_values("_r", kind="stable")
        queues[spk] = list(g.index)
    picked, rnd = [], 0
    while len(picked) < n and any(rnd < len(q) for q in queues.values()):
        for spk in sorted(queues):
            if len(picked) < n and rnd < len(queues[spk]):
                picked.append(queues[spk][rnd])
        rnd += 1
    sel = idx.loc[picked].sort_values("id").reset_index(drop=True)
    sel["tts_text"] = [tts_text(t) for t in sel["text"]]
    return sel[["id", "speaker", "chapter", "duration_s", "text", "tts_text", "path"]]


def _selftest() -> None:
    callers = emergency_callers()
    assert len(callers) >= 115, len(callers)
    assert len({c["text"] for c in callers}) == len(callers), "duplicate caller text"
    assert find_keywords("He was shot, gunshot") == ["shot"]
    assert find_keywords("I fell and I can't get up.") == ["fell", "cant get up"]
    assert find_keywords("There are fireworks") == []
    for c in callers:
        if c["category"] == "emergency":
            assert c["keywords"], f"emergency without keyword: {c['text']}"
    used = {k for c in callers for k in c["keywords"]}
    missing = set(KEYWORDS) - used
    assert not missing, f"keywords never used: {missing}"
    assert tts_text("I SAID I'LL GO") == "I said I'll go."
    cats = pd.Series([c["category"] for c in callers]).value_counts().to_dict()
    print("callers:", len(callers), cats)
    print("keyword counts:", pd.Series([k for c in callers for k in c["keywords"]]).value_counts().to_dict())


if __name__ == "__main__":
    _selftest()
