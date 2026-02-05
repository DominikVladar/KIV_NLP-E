#!/usr/bin/env python3
"""
OpenRouter mini-utility for checking key status.

Minimal usage:
  No packages needed, just Python 3.8+.
  Put API key in `OPENROUTER_API_KEY` environment variable and run script without arguments:
  `python3 openrouter_key_info.py`

Explanation:
  This script simply wraps the `/key` endpoint of the OpenRouter API with additional QoL.
  [Official docs here.](https://openrouter.ai/docs/api-reference/api-keys/get-current-key)

Optional Python dependencies:
  `dotenv` - load API key from `.env` file in case your project uses that
  `rich`   - to get styled outputs

Optional settings:
  The fetches all information from environment variables. I recommend using .env to inject global
  environment variables on a per-project basis. (Requires `dotenv` package to be installed.)
  - `OPENROUTER_API_KEY`: default variable to check for key
  - `OPENROUTER_API_KEY_VAR_NAME`: variable with name of api key variable for compatibility

Author: Jakub Hejman <hejmanj@kiv.zcu.cz>
"""

# does nothing if imported. If you want the info in your code, just use the endpoint as below.
if __name__ == "__main__":
    # import requests # i prefer like requests, but they are not built-in
    import urllib.request
    from datetime import datetime, timezone, timedelta
    import json
    import sys
    import os

    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass

    try:
        from rich.console import Console
        console = Console(highlight=False)

        def m_print(*args, **kwargs):
            console.print(*args, **kwargs)

    except ImportError:
        import re
        def m_print(*args, **kwargs):
            # remove rich console markup
            # (has false positives, since it does not check for keyword validity or mismatched tags)
            args = [re.sub(r"(?<!\\)\[.+?\]", "", a) for a in args]
            # and unescape the markup too
            args = [a.replace("\\[", "[") for a in args]
            print(*args, **kwargs)

    base_url = "https://openrouter.ai/api/v1/"
    key_location = os.environ.get("OPENROUTER_API_KEY_VAR_NAME", "OPENROUTER_API_KEY")
    api_key = os.environ.get(key_location, "")
    headers = {"Authorization": f"Bearer {api_key}"}

    def fmt_usd(dollars: float):
        return f"${dollars:,.3f}"

    if not api_key:
        m_print("[red3 bold]No key found, are you sure the environment is configured? (And dotenv installed?)")
        sys.exit(1)

    try:
        req = urllib.request.Request(base_url + "key", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read()).get("data", {})
    except Exception as ex:
        m_print("[red3 bold]Communication error!")
        m_print(ex)
        sys.exit(1)

    api_key_label = data.get("label", "Unknown API key")
    is_provisioning_key = data.get("is_provisioning_key", False)

    m_print(f"[green4 bold]API key: {api_key_label}[/] (From environment variable {key_location})")
    m_print()

    if is_provisioning_key:
        m_print("This is a provisioning key, which means it can generate more keys.")
        m_print("[red bold]Only ADMINS should have these and they cannot be used to make API calls.")
        
        try:
            req = urllib.request.Request(base_url + "credits", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            sys.exit() # if the additional request fails, don't bother the user
        
        total_credits = data.get("total_credits", 0.0)
        total_usage = data.get("total_usage", 0.0)
        m_print(f"Organization: [yellow]{fmt_usd(total_credits)}[/] available, [blue]{fmt_usd(total_usage)}[/] used")

    else:
        is_free_tier = data.get("is_free_tier", False)
        
        usage = data.get("usage", 0) # in $
        usage_daily = data.get("usage_daily", 0)
        usage_weekly = data.get("usage_weekly", 0)
        usage_monthly = data.get("usage_monthly", 0)
        
        limit = data.get("limit", 0) # in $
        limit_remaining = data.get("limit_remaining", 0)
        limit_reset = data.get("limit_reset", "") # weekly, monthly, etc

        expires_at = data.get("expires_at", "") # full ISO timestamp such as 2025-12-18T09:09:00.001+00:00
        
        # I don't think users care about BYOK, but the relevant keys would be:
        # "include_byok_in_limit": True, "byok_usage": 0, "byok_usage_daily": 0,
        # "byok_usage_weekly": 0, "byok_usage_monthly": 0

        if is_free_tier:
            m_print("This is a free tier API key and only free endpoints (with lower rate limits) can be used!")

        else:
            pct = limit_remaining / limit * 100 if limit else 0
            color = ('red' if pct < 20 else 'yellow' if pct < 50 else 'green') + " bold"
            remaining_print = f"[{color}]{fmt_usd(limit_remaining)}[/]"

            m_print(f"Current quota left/total: {remaining_print} / [bold]{fmt_usd(limit)}[/] ({pct:.0f}%)")
            
            if limit_reset:
                extra_info = " on Monday" if limit_reset == "weekly" else ""
                m_print(f"Quota resets {limit_reset} for this key. Resets at midnight UTC{extra_info}.")
            
            m_print("\nUsage:")
            m_print(f"today      {fmt_usd(usage_daily)}")
            m_print(f"this week  {fmt_usd(usage_weekly)}")
            m_print(f"this month {fmt_usd(usage_monthly)}")
            m_print(f"forever    {fmt_usd(usage)}")

        if expires_at:
            try:
                date = datetime.fromisoformat(expires_at)
            except ValueError:
                sys.exit() # if its malformed, just ignore it
            
            pretty_date = datetime(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=date.hour,
                minute=date.minute,
                tzinfo=timezone.utc # copy as correct timezone
            ).astimezone() # localize

            until = date - datetime.now(timezone.utc)
            pretty_until = timedelta(
                seconds=round(until.total_seconds())
            )

            m_print(f"\nKey expires on [tan]{pretty_date}[/] -> in {pretty_until}")
