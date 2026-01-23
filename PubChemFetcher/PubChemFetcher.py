# This program runs an interactive GUI that allows the user to search for a chemical compound via PubChem
# and retrieve some basic data on it. Python launches this program in a separate window as well, making it
# easier to use than using pycharm for the search function.

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pubchempy as pcp
import requests
import tkinter as tk
from tkinter import messagebox, scrolledtext

def fetch_compound_info():
    compound_name = entry.get().strip()
    if not compound_name:
        messagebox.showwarning("Input Error", "Please enter a compound name.")
        return

    try:
        compounds = pcp.get_compounds(compound_name, "name")

        if not compounds:
            messagebox.showinfo("Not Found", f"No compound found for: {compound_name}")
            return

        c = compounds[0]

        # Use modern smiles field instead of deprecated isomeric_smiles
        info = (
            f"Compound: {compound_name}\n"
            f"{'-'*50}\n"
            f"Molecular Formula : {c.molecular_formula}\n"
            f"Molecular Weight  : {c.molecular_weight}\n"
            f"SMILES            : {c.smiles}\n"
            f"Canonical SMILES  : {c.canonical_smiles}\n"
            f"IUPAC Name        : {c.iupac_name}\n"
            f"H-Bond Donors     : {c.h_bond_donor_count}\n"
            f"H-Bond Acceptors  : {c.h_bond_acceptor_count}\n"
            f"Rotatable Bonds   : {c.rotatable_bond_count}\n"
        )

        # Create a new window for results
        result_window = tk.Toplevel(root)
        result_window.title(f"{compound_name} - PubChem Info")

        text_area = scrolledtext.ScrolledText(result_window, width=60, height=20, wrap=tk.WORD)
        text_area.pack(padx=10, pady=10)
        text_area.insert(tk.END, info)
        text_area.configure(state="disabled")  # read-only

    except requests.exceptions.SSLError as e:
        messagebox.showerror("SSL Error", str(e))

    except requests.exceptions.ConnectionError as e:
        messagebox.showerror("Network Error", str(e))

    except Exception as e:
        messagebox.showerror("Unexpected Error", f"{type(e).__name__}: {e}")


# ---- GUI Setup ----
root = tk.Tk()
root.title("PubChem Compound Lookup")

tk.Label(root, text="Enter a compound name:").pack(padx=10, pady=(10, 0))

entry = tk.Entry(root, width=40)
entry.pack(padx=10, pady=5)

search_button = tk.Button(root, text="Search PubChem", command=fetch_compound_info)
search_button.pack(pady=10)

root.mainloop()
