"""One-time, interactive setup of Phase-2 credentials in the OS keychain (nothing is committed).

Run:  python setup_credentials.py

Stores, via the `keyring` library (macOS Keychain / Windows Credential Locker / libsecret):
  * the **X API bearer token** (for the low-latency tweet feed), and
  * the **Polymarket trading-wallet private key** (for SELL execution).

Secrets are read only at use time and never logged. Use a **dedicated low-balance trading wallet**
for Polymarket so the key's blast radius is bounded. Requires `pip install -r requirements-live.txt`.
"""
import getpass
import sys

sys.path.insert(0, "src")


def main() -> None:
    try:
        import keyring  # noqa: F401
    except ImportError:
        print("Le module `keyring` est requis. Lance d'abord :  pip install -r requirements-live.txt")
        sys.exit(1)

    from tweetanalyst import execution as X
    from tweetanalyst import sources as S

    print("=== Configuration des identifiants (stockés dans le trousseau OS, jamais committés) ===\n")

    # --- X API bearer token ---
    if input("Configurer le token X API ? [o/N] ").strip().lower() in ("o", "y", "oui", "yes"):
        tok = getpass.getpass("  Token bearer X API (saisie masquée) : ").strip()
        if tok:
            S.store_x_token(tok)
            print("  ✅ Token X enregistré (service:", S._X_KEYRING_SERVICE, ").")
        else:
            print("  (ignoré — vide)")

    # --- Polymarket private key ---
    if input("\nConfigurer la clé privée du wallet de trading Polymarket ? [o/N] ").strip().lower() \
            in ("o", "y", "oui", "yes"):
        print("  ⚠️  Utilise un wallet DÉDIÉ à faible solde. La clé sert à signer les ordres de VENTE.")
        key = getpass.getpass("  Clé privée (saisie masquée) : ").strip()
        if key:
            X.store_private_key(key)
            print("  ✅ Clé privée enregistrée (service:", X._KEYRING_SERVICE, ").")
        else:
            print("  (ignoré — vide)")

    print("\nTerminé. Pour ACTIVER (séparément, quand tu es prêt) :")
    print("  • Feed X         : sources.LIVE_X_ENABLED = True")
    print("  • Exécution vente: execution.EXECUTION_ENABLED = True  (+ confirm=True par ordre)")
    print("Rien n'est actif tant que ces drapeaux restent à False.")


if __name__ == "__main__":
    main()
