import sqlite3
import matplotlib.pyplot as plt
import collections
import numpy as np
import re
from pathlib import Path
from itertools import groupby
import time

def create_graph(results: dict, thresholds: list[float]):
    for domain, sizes in results.items():
        for size, rows in sizes.items():
            if size == 5 and domain == "simpleExreg":
                continue
            fig = plt.figure(figsize=(16,9), dpi=300)
            ax = plt.axes()
            fig.add_axes(ax)
            ax.tick_params(axis='both', which='major', labelsize=20)
            y = np.asarray(rows['tp'])
            y_err = np.asarray(rows['error_tp'])
            ax.plot(thresholds, rows['tp'] , label='true positives', color="tab:blue", linewidth=4)
            ax.fill_between(thresholds, y - y_err, y+ y_err, alpha=0.5)
            ax.set_title(f"chunk size: {size}", fontsize=20)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("threshold", fontsize=16)
            ax.set_ylabel("true positvies", fontsize=16)
            ax2 = ax.twinx()
            y = np.asarray(rows['fp'])
            y_err = np.asarray(rows['error_fp'])
            ax2.tick_params(axis='both', which='major', labelsize=20)
            ax2.set_ylabel("false positives", fontsize=16)
            ax2.plot(thresholds, rows['fp'] , label='false positives', color="tab:red", linewidth=4)
            ax2.fill_between(thresholds, y - y_err, y+ y_err, alpha=0.5, facecolor="tab:red")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0, -0.2), prop={'size': 16}, loc='lower left')
            fig.savefig(f"tp_fp_{domain}_{size}.png", bbox_inches="tight")
            plt.close()

def show_results():
    with sqlite3.connect('results.db') as con:
        con.enable_load_extension(True)
        con.row_factory = sqlite3.Row
        con.execute("select load_extension('./stats.dll')")
        
        cur = con.execute("""
        select  threshold,
                domain_name,
                split_size,
                avg(true_positives) mean_tp,
                stddev_pop(true_positives) / sqrt(count(*)) error_tp,
                avg(false_positives) mean_fp,
                stddev_pop(false_positives) / sqrt(count(*)) error_fp,
                with_kl from result
        where with_kl = 0
        group by with_kl, domain_name, threshold, split_size
        ORDER BY split_size
                    """)
        rows = cur.fetchall()
        domains = set(map(lambda r: r['domain_name'], rows))
        thresholds = list(sorted(set(map(lambda r: r['threshold'], rows))))
        split_sizes = set(map(lambda r: r['split_size'], rows))
        results = dict()
        for domain in domains:
            results[domain] = dict()
            for size in split_sizes:
                results[domain][size] = { 'tp': [], 'error_tp': [], 'fp': [], 'error_fp': [] }
        for row in rows:
            domain = row['domain_name']
            size = row['split_size']
            results[domain][size]['tp'].append(row['mean_tp'])
            results[domain][size]['fp'].append(row['mean_fp'])
            results[domain][size]['error_tp'].append(row['error_tp'])
            results[domain][size]['error_fp'].append(row['error_fp'])

        create_graph(results, thresholds)
domain_regexes = [
        ('generate_url_scheme_1',r'[abc][abc][ab][ab]-domain\.com'),
        ('generate_url_scheme_2',r'[abcd][abcd][abcd][abcd]-domain\.com'),
        #('simpleExreg', r'[0123456789][0123456789][0123456789]([abc][abc]|[abc][abc][abc]|[abc][abc][abc][abc]|[abc][abc][abc][abc][abc])(|[xyz]|[xyz][xyz]|[xyz][xyz][xyz]|[xyz][xyz][xyz][xyz])\.(evil|knievel)\.com'),
        #('statefulExreg', (r'[0123456789][0123456789][0123456789][0123456789][0123456789]\.evil\.([abc]|[abc][abc])\.com',
        #r'[0123456789][0123456789][0123456789][0123456789][0123456789]\.knievel\.([abc][abc][abc]|[abc][abc][abc][abc])\.com')),
        #('statefulExregSameLength',(r'[0123456789][0123456789][0123456789][0123456789][0123456789]\.evil\.[abc][abc][abc][abc]\.com',
        #r'[0123456789][0123456789][0123456789]\.knievel\.[abc][abc][abc]\.com'))
        ]

def split_into_atoms(regex: str, size: int):
    splits = []
    current = ''
    current_size = 0
    in_bracket = False
    for c in regex:
        if c == '[':
            in_bracket = True
            current += c
        elif in_bracket and c == ']':
            current_size += 1
            current += c
            in_bracket = False
        elif in_bracket:
            current += c
        elif c == '\\':
            current += c
        else:
            current += c
            current_size += 1

        if current_size == size:
            splits.append(current)
            current = ''
            current_size = 0
    if current != '':
        splits.append(current)
    return splits
 

def regex_show_distances():
    with sqlite3.connect('results.db') as con:
        con.enable_load_extension(True)
        con.row_factory = sqlite3.Row
        con.execute("select load_extension('./stats.dll')")
        con.execute("select load_extension('./fuzzy.dll')")
        con.set_trace_callback(print)
        for domain_name, domain in domain_regexes:
            for split_size in range(2,6):
                fig = plt.figure(figsize=(16,12), dpi=300)
                ax = plt.axes()
                fig.add_axes(ax)
                split_domain = ''.join(f"({s})" for s in split_into_atoms(domain, split_size))
                query = (r"""
                select 
                    threshold, 
                    split_size, 
                    avg(true_positives) as true_positives, 
                    stddev_pop(true_positives)/sqrt(count(*)) as tp_error,
                    regex, 
                    avg(fuzzy_damlev(regex, ?)) as mean_dst,
                    stddev_pop(fuzzy_damlev(regex, ?)) / sqrt(count(*)) as dst_error
                from result
                where domain_name = ? and split_size = ?
                group by threshold, domain_name, split_size
            """)
                cur = con.execute(query, (split_domain, split_domain, domain_name, split_size) ) 
            
                results = cur.fetchall()
                thresholds = list(r['threshold'] for r in results)
                tp_error = np.asarray([r['tp_error'] for r in results])
                true_positives = np.asarray([r['true_positives'] for r in results])
                distance = np.asarray([r['mean_dst'] for r in results])
                error = np.asarray([r['dst_error'] for r in results])
                ax.set_xlabel('thresholds', fontsize=16)
                ax.set_ylabel('Damlev-Levenshtein distance', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax2 = ax.twinx()
                ax.plot(thresholds, distance, label='edit distance', linewidth=4)
                ax2.plot(thresholds, true_positives, label='true_positives', color="tab:cyan", linewidth=4)
                ax2.set_ylabel('true positives', fontsize=16)
                ax.fill_between(thresholds, distance - error, distance + error, alpha=0.5)
                ax2.fill_between(thresholds, true_positives - tp_error, true_positives + tp_error, alpha=0.5, color="tab:cyan")
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0,-0.2), loc='lower left', prop={'size': 16})
                ax.set_title(f"chunk size: {split_size}", fontsize=20)
                fig.savefig(f'dst_{domain_name}_{split_size}.png', bbox_inches='tight')



if __name__ == "__main__":
    show_results()
    regex_show_distances()
