# 🏛️ Reclaiming Our Atlantic Destiny (R.O.A.D.)
## Handwritten Text Recognition Challenge — Barbados Archives

> *Can you identify handwritten words in historic and culturally significant scripts from the archives of Barbados?*

---

## 📜 Background

Barbados' colonial-era history is documented in **thousands of handwritten pages** — deeds, wills, estate inventories, census records — preserved and digitised through the **Reclaiming Our Atlantic Destiny (R.O.A.D.) Programme**, powered by **GovTech Barbados**.

These documents hold invaluable insights into the lives, economies, and histories of Caribbean people. But many of these records are difficult to read:

- Faded ink
- Degraded pages
- Unfamiliar handwriting styles

Together, these factors make manual transcription **slow, expensive, and inaccessible at scale**.

---

## 🎯 Your Mission

Build a machine learning model that can **automatically recognise and transcribe historical handwritten text** from scanned images provided by R.O.A.D.

Think of it as building a **digital historian**: your model should convert irregular, handwritten historical records into clean, machine-readable text that can be used for research, storytelling, and digital preservation.

This challenge is about more than just technical transcription. A strong model will unlock **faster, more scalable digitisation** of archival data, transforming the way researchers, historians, and communities interact with their past.

> The impact goes beyond Barbados — the winning solution could serve as a **blueprint for digitising other dispersed archives across the Commonwealth.**

---

## 🌍 About Reclaiming Our Atlantic Destiny (R.O.A.D.)

**Reclaiming Our Atlantic Destiny (R.O.A.D.)** is a Barbados-led transformative initiative built on two mutually reinforcing pillars:

1. **Large-scale preservation and digitisation** of Barbados' incomparable archives
2. **The establishment of the Barbados Heritage District** — a world-class cultural heritage precinct

Together, these pillars respond to a defining national and global opportunity: to safeguard **tens of millions of pages** of records documenting the development of Trans-Atlantic slave societies, the lives of the enslaved, the liberated, and their descendants — while opening these histories to research, education, and discovery.

This work transforms fragile and irreplaceable records into a **secure, searchable, and enduring resource** for Barbadians, the diaspora, and the wider world. In doing so, the Programme is:

- Building new technical capacity
- Strengthening Barbados' cultural heritage infrastructure
- Positioning the country as a **global centre for knowledge, innovation, and sustainable heritage-led development**

---

## 📊 Evaluation

This challenge uses **multi-metric evaluation** based on two error metrics:

| Metric | Weighting |
|---|---|
| **WER** (weighted Word Error Rate) | 0.5 |
| **CER** (weighted Character Error Rate) | 0.5 |

The final leaderboard score is the **weighted mean** of these two metrics.

- **WER** measures errors at the *word* level.
- **CER** measures errors at the *character* level.

Together, they capture both word-level transcription quality and finer spelling or character-level mistakes — especially important for text/speech recognition tasks across languages with different spelling and word-boundary patterns.

### ⚖️ Weighting Notes

- **Longer reference transcriptions are weighted more heavily**, so errors on longer samples contribute more to the final score than errors on very short samples.
- Submissions **must include predictions for all required IDs**.
- **Missing predictions, empty predictions, or invalid text values** will be penalised as incorrect.

---

*Help preserve the past. Build the future.* 🇧🇧