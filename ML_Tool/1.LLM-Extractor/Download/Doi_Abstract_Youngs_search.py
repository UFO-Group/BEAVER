#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python Doi_Abstract_Youngs_search.py


import requests
import pandas as pd
import time
import re
import os


# ==== Clean HTML Tags ====
def clean_html(raw_html: str) -> str:
    cleanr = re.compile(r"<.*?>")
    return re.sub(cleanr, "", raw_html).strip()


# ==== Construct Query String ====
def build_query(words):

    query_parts = []
    for word in words:
        if isinstance(word, list):
            query_parts.append("(" + "+OR+".join(word) + ")")
        else:
            query_parts.append(word)
    return "+".join(query_parts)


# ==== Main Crawler Logic ====
def fetch_dois_with_logic(
    and_words,
    or_sets=None,
    not_words=None,
    total=3000,
    per_page=500,
    from_pub_date="2010-01-01",
):

    results = []
    or_sets = or_sets or [[]]
    headers = {"User-Agent": "MyResearchBot/1.0 (mailto:your_email)"}

    for or_words in or_sets:
        all_words = and_words + or_words
        query_string = build_query(all_words)
        print(f"\n🔍 Query Keyword Combination: {all_words}")

        seen_dois = set()
        current_total = 0
        empty_count = 0

        for offset in range(0, total, per_page):
            if current_total >= total:
                print("📈 Reached maximum fetch limit, terminating this combination early.")
                break

            print(f"→ Fetching records: {offset + 1} to {offset + per_page}...")
            url = (
                "https://api.crossref.org/works"
                f"?query.bibliographic={query_string}"
                f"&rows={per_page}"
                f"&offset={offset}"
                f"&filter=type:journal-article,from-pub-date:{from_pub_date}"
            )

            data = None
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code != 200:
                        print(f"❌ Request failed (Status Code {response.status_code}), skipping page")
                        break
                    data = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Request exception: {e}, retrying {attempt + 1}/3")
                    time.sleep(2)

            if data is None:
                empty_count += 1
                if empty_count >= 3:
                    print("🛑 3 consecutive failures, terminating this combination.")
                    break
                continue

            items = data.get("message", {}).get("items", [])
            if not items:
                empty_count += 1
                if empty_count >= 3:
                    print("🛑 3 consecutive empty pages, terminating this combination.")
                    break
                continue
            else:
                empty_count = 0

            new_entries = 0
            for item in items:
                doi = item.get("DOI", "")
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)

                title = item.get("title", [""])[0]
                abstract_raw = item.get("abstract", "")
                abstract = clean_html(abstract_raw) if abstract_raw else ""

                # NOT Filtering (Optional)
                if not_words:
                    lower_text = (title + " " + abstract).lower()
                    if any(nw.lower() in lower_text for nw in not_words):
                        continue

                results.append(
                    {
                        "Title": title,
                        "DOI": doi,
                        "Abstract": abstract,
                        "QueryWords": " ".join(
                            " / ".join(w) if isinstance(w, list) else w
                            for w in all_words
                        ),
                    }
                )
                new_entries += 1

            if new_entries == 0:
                print("🛑 All items on this page are duplicates, no new data, terminating this combination.")
                break

            current_total += new_entries
            time.sleep(2)

    return results


def main():
    # ==== Material Keywords ====
    names = [
        "polymer", "copolymer", "blend", "biopolymer",
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "polymethyl methacrylate",
        "polyurethane", "polyamide", "nylon", "polyvinyl chloride",
        "PVA", "polyvinyl alcohol", "polyacrylonitrile", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "polycarbonate", "polybutylene succinate",
        "PGA", "poly(γ-glutamic acid)",  "polyethylene", "polypropylene", "polyester",
        "GelMA", "gelatin methacrylate", "Gelatin", "Collagen", "Chitosan",
        "Sodium alginate", "alginate", "Cellulose", "Hyaluronic acid", 
        "silk fibroin", "polyethylene glycol", "polydopamine", "Polyacrylamide"
    ]

    # ==== Mechanical Properties Keywords (AND Condition) ====
    base_and = [
        "young's modulus", "modulus of elasticity", "elastic modulus", "tensile modulus", "tensile strength",
        "mechanical modulus", "elongation at break", "compressive strength", "moduli", "stiffness",
        "flexural modulus", "bending modulus", "impact strength",
        "impact resistance", "stress", "strain",
        "mechanical properties", "mechanical performance", "dynamic mechanical", 
        "fracture toughness", "toughness",
        "hardness shore", "flexibility", "rigidity",
        "compression strength", "creep resistance"
    ]

    # ==== Save Path ====
    save_dir = r"your_YoungsModulus_DOI"
    os.makedirs(save_dir, exist_ok=True)

    # ==== Fetch and Save by Keyword Combination ====
    file_counter = 1
    global_seen_dois = set()

    for name in names:
        for kw in base_and:
            # kw are strings here, logic preserves compatibility for list cases
            if isinstance(kw, list):
                and_group = [name] + kw
                kw_str = "_".join(kw)
            else:
                and_group = [name, kw]
                kw_str = kw

            print("\n" + "=" * 60)
            print(f"📚 Combination: {name} + {kw_str}")
            print("=" * 60)

            results = fetch_dois_with_logic(
                and_words=and_group,
                total=3000,
                per_page=500,
            )

            # Double deduplication: Current group + Global
            seen_in_group = set()
            unique_results = []
            for item in results:
                doi = item["DOI"]
                if doi and doi not in global_seen_dois and doi not in seen_in_group:
                    seen_in_group.add(doi)
                    global_seen_dois.add(doi)
                    unique_results.append(item)

            if unique_results:
                df = pd.DataFrame(unique_results)
                mat_str = name.replace(" ", "_")
                safe_kw_str = kw_str.replace(" ", "_")
                filename = os.path.join(
                    save_dir, f"{file_counter:04d}_{mat_str}_{safe_kw_str}.csv"
                )
                df.to_csv(filename, index=False, encoding="utf-8-sig")
                print(f"💾 File saved: {filename}, Total {len(df)} references")
                file_counter += 1
            else:
                print(f"⚠️ {name} + {kw_str} No new references, skipping.")


if __name__ == "__main__":
    main()