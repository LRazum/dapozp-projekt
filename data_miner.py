from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


WB_API = "https://api.worldbank.org/v2"
RC_API = "https://restcountries.com/v3.1/all"
CACHE_DIR = Path(".cache_countries")
DATE_RANGE = "2015:2024"
TIMEOUT = 60
MAX_RETRIES = 3
USER_AGENT = "CountriesDatasetBuilder/1.0 (educational, course project)"

# Scope projekta - ignoriramo sve izvan ovih 209 ISO3 kodova
VALID_COUNTRIES = frozenset({
    "AFG","ALB","DZA","AND","AGO","ATG","ARG","ARM","ABW","AUS","AUT","AZE",
    "BHS","BHR","BGD","BRB","BLR","BEL","BLZ","BEN","BMU","BTN","BOL","BIH",
    "BWA","BRA","VGB","BRN","BGR","BFA","BDI","CPV","KHM","CMR","CAN","CYM",
    "CAF","TCD","CHI","CHL","CHN","COL","COM","COD","COG","CRI","CIV","HRV",
    "CUB","CUW","CYP","CZE","DNK","DJI","DMA","DOM","ECU","EGY","SLV","GNQ",
    "ERI","EST","SWZ","FRO","FJI","FIN","FRA","PYF","GAB","GMB","GEO",
    "DEU","GHA","GRC","GRL","GRD","GTM","GIN","GNB","GUY","HTI","HND","HKG",
    "HUN","ISL","IND","IDN","IRN","IRQ","IRL","IMN","ISR","ITA","JAM","JPN",
    "JOR","KAZ","KEN","KIR","KOR","XKX","KWT","KGZ","LAO","LVA","LBN",
    "LSO","LBR","LBY","LIE","LTU","LUX","MAC","MDG","MWI","MYS","MDV","MLI",
    "MLT","MHL","MRT","MUS","MEX","FSM","MDA","MCO","MNG","MNE","MAR","MOZ",
    "MMR","NAM","NRU","NPL","NLD","NCL","NZL","NIC","NER","NGA","MKD","NOR",
    "OMN","PAK","PLW","PAN","PNG","PRY","PER","PHL","POL","PRT","PRI","QAT",
    "ROU","RUS","RWA","WSM","SMR","STP","SAU","SEN","SRB","SYC","SLE","SGP",
    "SXM","SVK","SVN","SLB","SOM","ZAF","SSD","ESP","LKA","KNA","LCA","VCT",
    "SDN","SUR","SWE","CHE","SYR","TJK","TZA","THA","TLS","TGO","TON","TTO",
    "TUN","TUR","TKM","TCA","TUV","UGA","UKR","ARE","GBR","USA","URY","UZB",
    "VUT","VNM","VIR","PSE","YEM","ZMB","ZWE",
})

WB_INDICATORS: dict[str, str] = {
    "NY.GDP.PCAP.PP.CD":  "gdp_per_capita_ppp",
    "SL.UEM.TOTL.ZS":     "unemployment_rate",
    "NV.IND.TOTL.ZS":     "industry_pct_gdp",
    "NV.SRV.TOTL.ZS":     "services_pct_gdp",
    "NV.AGR.TOTL.ZS":     "agriculture_pct_gdp",
    "GB.XPD.RSDV.GD.ZS":  "rd_expenditure_pct_gdp",
    "FP.CPI.TOTL.ZG":     "inflation_pct",
    "SP.POP.TOTL":        "population_total",
    "EN.POP.DNST":        "population_density",
    "SP.URB.TOTL.IN.ZS":  "urban_population_pct",
    "SP.POP.GROW":        "population_growth_pct",
    "SP.DYN.TFRT.IN":     "fertility_rate",
    "SP.POP.65UP.TO.ZS":  "pop_65_plus_pct",
    "SP.DYN.LE00.IN":     "life_expectancy",
    "SP.DYN.IMRT.IN":     "infant_mortality",
    "SH.XPD.CHEX.GD.ZS":  "health_expenditure_pct_gdp",
    "SE.XPD.TOTL.GD.ZS":  "education_expenditure_pct_gdp",
    "IT.NET.USER.ZS":     "internet_users_pct",
    "EG.ELC.ACCS.ZS":     "electricity_access_pct",
}

INDICATOR_COLS = list(WB_INDICATORS.values())


ISO3_TO_SUBREGION: dict[str, str] = {}
for codes, name in [
    (['DZA','EGY','LBY','MAR','SDN','TUN'],                    'Northern Africa'),
    (['BDI','COM','DJI','ERI','ETH','KEN','MDG','MWI','MUS',
      'MOZ','RWA','SYC','SOM','SSD','TZA','UGA','ZMB','ZWE'],  'Eastern Africa'),
    (['AGO','CMR','CAF','TCD','COD','COG','GNQ','GAB','STP'],  'Middle Africa'),
    (['BWA','SWZ','LSO','NAM','ZAF'],                          'Southern Africa'),
    (['BEN','BFA','CPV','CIV','GMB','GHA','GIN','GNB','LBR',
      'MLI','MRT','NER','NGA','SEN','SLE','TGO'],              'Western Africa'),
    (['BMU','CAN','GRL','USA'],                                'Northern America'),
    (['ATG','ABW','BHS','BRB','VGB','CYM','CUB','CUW','DMA',
      'DOM','GRD','HTI','JAM','PRI','SXM','KNA','LCA','VCT',
      'TTO','TCA','VIR'],                                      'Caribbean'),
    (['BLZ','CRI','SLV','GTM','HND','MEX','NIC','PAN'],        'Central America'),
    (['ARG','BOL','BRA','CHL','COL','ECU','GUY','PRY','PER',
      'SUR','URY','VEN'],                                      'South America'),
    (['KAZ','KGZ','TJK','TKM','UZB'],                          'Central Asia'),
    (['CHN','HKG','MAC','JPN','MNG','KOR','PRK'],              'Eastern Asia'),
    (['BRN','KHM','IDN','LAO','MYS','MMR','PHL','SGP','THA',
      'TLS','VNM'],                                            'South-eastern Asia'),
    (['AFG','BGD','BTN','IND','IRN','MDV','NPL','PAK','LKA'],  'Southern Asia'),
    (['ARM','AZE','BHR','CYP','GEO','IRQ','ISR','JOR','KWT',
      'LBN','OMN','PSE','QAT','SAU','SYR','TUR','ARE','YEM'],  'Western Asia'),
    (['BLR','BGR','CZE','HUN','MDA','POL','ROU','RUS','SVK',
      'UKR'],                                                  'Eastern Europe'),
    (['CHI','DNK','EST','FRO','FIN','ISL','IRL','IMN','LVA',
      'LTU','NOR','SWE','GBR'],                                'Northern Europe'),
    (['ALB','AND','BIH','HRV','GRC','ITA','XKX','MLT','MNE',
      'MKD','PRT','SMR','SRB','SVN','ESP'],                    'Southern Europe'),
    (['AUT','BEL','FRA','DEU','LIE','LUX','MCO','NLD','CHE'],  'Western Europe'),
    (['AUS','NZL'],                                            'Australia and New Zealand'),
    (['FJI','NCL','PNG','SLB','VUT'],                          'Melanesia'),
    (['KIR','MHL','FSM','NRU','PLW'],                          'Micronesia'),
    (['PYF','WSM','TON','TUV'],                                'Polynesia'),
]:
    for code in codes:
        ISO3_TO_SUBREGION[code] = name

SUBREGION_TO_CONTINENT: dict[str, str] = {
    'Northern Africa': 'Africa', 'Eastern Africa': 'Africa',
    'Middle Africa': 'Africa',   'Southern Africa': 'Africa',
    'Western Africa': 'Africa',
    'Northern America': 'Americas', 'Caribbean': 'Americas',
    'Central America': 'Americas',  'South America': 'Americas',
    'Central Asia': 'Asia',         'Eastern Asia': 'Asia',
    'South-eastern Asia': 'Asia',   'Southern Asia': 'Asia',
    'Western Asia': 'Asia',
    'Eastern Europe': 'Europe',     'Northern Europe': 'Europe',
    'Southern Europe': 'Europe',    'Western Europe': 'Europe',
    'Australia and New Zealand': 'Oceania', 'Melanesia': 'Oceania',
    'Micronesia': 'Oceania',                'Polynesia': 'Oceania',
}

UN_NON_MEMBERS = frozenset({
    'ABW','BMU','VGB','CYM','CHI','CUW','FRO','PYF','GRL','HKG','IMN','XKX',
    'MAC','NCL','PRI','SXM','TCA','VIR','PSE',
})

EU_MEMBERS = frozenset({
    'AUT','BEL','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC',
    'HUN','IRL','ITA','LVA','LTU','LUX','MLT','NLD','POL','PRT','ROU','SVK',
    'SVN','ESP','SWE',
})

OECD_MEMBERS = frozenset({
    'AUS','AUT','BEL','CAN','CHL','COL','CRI','CZE','DNK','EST','FIN','FRA',
    'DEU','GRC','HUN','ISL','IRL','ISR','ITA','JPN','KOR','LVA','LTU','LUX',
    'MEX','NLD','NZL','NOR','POL','PRT','SVK','SVN','ESP','SWE','CHE','TUR',
    'GBR','USA',
})

NATO_MEMBERS = frozenset({
    'USA','GBR','FRA','DEU','ITA','ESP','CAN','BEL','NLD','NOR','DNK','ISL',
    'LUX','PRT','GRC','TUR','ALB','BGR','HRV','CZE','EST','HUN','LVA','LTU',
    'MNE','MKD','POL','ROU','SVK','SVN','FIN','SWE',
})

G20_MEMBERS = frozenset({
    'ARG','AUS','BRA','CAN','CHN','FRA','DEU','IND','IDN','ITA','JPN','KOR',
    'MEX','RUS','SAU','ZAF','TUR','GBR','USA',
})

COMMONWEALTH_MEMBERS = frozenset({
    'GBR','ATG','AUS','BHS','BGD','BRB','BLZ','BWA','BRN','CMR','CAN','CYP',
    'DMA','SWZ','FJI','GAB','GMB','GHA','GRD','GUY','IND','JAM','KEN','KIR',
    'LSO','MWI','MYS','MDV','MLT','MUS','MOZ','NAM','NRU','NZL','NGA','PAK',
    'PNG','RWA','WSM','SYC','SLE','SGP','SLB','ZAF','LKA','KNA','LCA','VCT',
    'TZA','TGO','TON','TTO','TUV','UGA','VUT','ZMB',
})

OPEC_MEMBERS = frozenset({
    'DZA','COG','GNQ','GAB','IRN','IRQ','KWT','LBY','NGA','SAU','ARE','VEN',
})

LANGUAGE_FAMILY: dict[str, str] = {}
for family, codes in [
    ('Indo-European', [
        'ALB','AND','ARG','AUT','BEL','BLR','BOL','BIH','BRA','BGR','CAN','CHL',
        'COL','HRV','CUB','CZE','DNK','DOM','ECU','SLV','FRA','DEU','GRC','GTM',
        'GUY','HND','ISL','IRL','ITA','LVA','LIE','LTU','LUX','MKD','MEX','MDA',
        'MCO','MNE','NLD','NIC','NOR','PAN','PRY','PER','POL','PRT','ROU','RUS',
        'SMR','SRB','SVK','SVN','ESP','SWE','CHE','UKR','GBR','URY','VEN','XKX',
        'AUS','NZL','USA','SUR','BHS','BRB','BLZ','DMA','GRD','JAM','KNA','LCA',
        'VCT','TTO','ATG','VGB','CYM','TCA','BMU','VIR','PRI','ABW','CUW','SXM',
        'CHI','IMN','FRO','CYP','CRI',
        'AFG','IRN','TJK','BGD','IND','MDV','NPL','PAK','LKA','ARM',
    ]),
    ('Afro-Asiatic', [
        'DZA','BHR','EGY','IRQ','JOR','KWT','LBN','LBY','MAR','OMN','PSE','QAT',
        'SAU','SYR','TUN','ARE','YEM','ISR','SDN','DJI','SOM','ERI','MRT','COM',
        'MLT','ETH',
    ]),
    ('Sino-Tibetan',   ['CHN','HKG','MAC','MMR','BTN','SGP']),
    ('Niger-Congo', [
        'AGO','BEN','BWA','BFA','BDI','CMR','CAF','COD','COG','CIV','GNQ','SWZ',
        'GAB','GMB','GHA','GIN','GNB','KEN','LSO','LBR','MWI','MLI','MUS','MOZ',
        'NAM','NER','NGA','RWA','STP','SEN','SYC','SLE','ZAF','TGO','UGA','TZA',
        'ZMB','ZWE','CPV',
    ]),
    ('Nilo-Saharan',   ['TCD','SSD']),
    ('Austronesian', [
        'BRN','IDN','MYS','PHL','MDG','TLS','FJI','NCL','PNG','SLB','VUT','KIR',
        'MHL','FSM','NRU','PLW','PYF','WSM','TON','TUV',
    ]),
    ('Turkic',         ['TUR','AZE','KAZ','KGZ','TKM','UZB']),
    ('Uralic',         ['FIN','EST','HUN']),
    ('Austroasiatic',  ['VNM','KHM']),
    ('Tai-Kadai',      ['THA','LAO']),
    ('Japonic',        ['JPN']),
    ('Koreanic',       ['KOR','PRK']),
    ('Mongolic',       ['MNG']),
    ('Kartvelian',     ['GEO']),
    ('Eskimo-Aleut',   ['GRL']),
    ('Creole',         ['HTI']),
]:
    for c in codes:
        LANGUAGE_FAMILY[c] = family

# Ručni overrideovi jer vanjski API-ji često brljaju s otočnim državama i ekvatorom
SOUTHERN_HEMISPHERE = frozenset({
    'AGO','BWA','BDI','COM','COD','LSO','MDG','MWI','MOZ','NAM','RWA','SYC',
    'ZAF','SWZ','TZA','ZMB','ZWE','ARG','BOL','BRA','CHL','PRY','PER','URY',
    'AUS','NZL','FJI','NCL','PNG','SLB','VUT','WSM','TON','TUV','PYF','TLS','IDN',
})

ISLAND_NATION = frozenset({
    'ATG','AUS','ABW','BHS','BHR','BRB','BMU','VGB','CPV','CYM','CHI','COM',
    'CUB','CUW','CYP','DMA','DOM','FRO','FJI','PYF','GRD','GRL','HTI','ISL',
    'IDN','IRL','IMN','JAM','JPN','KIR','MDG','MDV','MLT','MHL','MUS','FSM',
    'NRU','NCL','NZL','PLW','PHL','PRI','WSM','STP','SGP','SXM','SLB','LKA',
    'KNA','LCA','VCT','TLS','TON','TTO','TCA','TUV','GBR','VUT','VIR',
})

LANDLOCKED = frozenset({
    'AFG','AND','ARM','AUT','AZE','BLR','BTN','BOL','BFA','BDI','CAF','TCD',
    'CZE','HUN','KAZ','XKX','KGZ','LAO','LSO','LIE','LUX','MWI','MLI','MDA',
    'MNG','NPL','NER','MKD','PRY','RWA','SMR','SRB','SVK','SSD','SWZ','CHE',
    'TJK','TKM','UGA','UZB','ZMB','ZWE',
})

# Hardkodirani fallbackovi za nestandardne teritorije koje API-ji ne vraćaju kako treba
CAPITAL_FALLBACKS = {
    'CHI': 'Saint Helier',
    'HKG': 'Hong Kong',
    'ISR': 'Jerusalem',
    'MAC': 'Macao',
    'PSE': 'Ramallah',
}

LAND_AREA_FALLBACKS = {
    'CHI': 194,
    'XKX': 10887,
}

TIER_SPECS = [
    ('gdp_per_capita_ppp',  'gdp_tier', [-1, 5_000, 15_000, 35_000, 1e9], ['Low', 'Lower-middle', 'Upper-middle', 'High']),
    ('population_total',    'population_tier', [-1, 1e6, 1e7, 5e7, 2e8, 2e10], ['Tiny', 'Small', 'Medium', 'Large', 'Mega']),
    ('population_density',  'density_tier', [-1, 30, 150, 500, 1e6], ['Sparse', 'Moderate', 'Dense', 'Very-dense']),
    ('pop_65_plus_pct',     'aging_tier', [-1, 5, 10, 15, 20, 100], ['Young', 'Maturing', 'Mature', 'Aged', 'Super-aged']),
    ('urban_population_pct','urbanization_tier', [-1, 30, 60, 85, 101], ['Rural', 'Mixed', 'Urban', 'Hyper-urban']),
    ('internet_users_pct',  'internet_tier', [-1, 30, 70, 90, 101], ['Low', 'Medium', 'High', 'Very-high']),
    ('life_expectancy',     'life_expectancy_tier', [-1, 60, 70, 78, 110], ['Low', 'Medium', 'High', 'Very-high']),
    ('fertility_rate',      'fertility_tier', [-1, 2.1, 3, 5, 10], ['Below-replacement', 'Replacement', 'High', 'Very-high']),
]

OUTPUT_COLS = [
    'iso3','country_name','capital',
    'continent','region','hemisphere','landlocked','island_nation','land_area_km2',
    'un_member','eu_member','oecd_member','nato_member','g20_member','commonwealth_member','opec_member',
    'primary_language_family',
    'income_group','gdp_per_capita_ppp','gdp_tier','unemployment_rate',
    'industry_pct_gdp','services_pct_gdp','agriculture_pct_gdp',
    'economic_structure','rd_expenditure_pct_gdp','inflation_pct',
    'population_total','population_tier','population_density','density_tier',
    'population_growth_pct','urban_population_pct','urbanization_tier',
    'fertility_rate','fertility_tier','pop_65_plus_pct','aging_tier',
    'life_expectancy','life_expectancy_tier','infant_mortality',
    'health_expenditure_pct_gdp','education_expenditure_pct_gdp',
    'internet_users_pct','internet_tier','electricity_access_pct',
    'imputed_fields_count',
]


def get_eco_structure(row: pd.Series) -> str:
    if row['agriculture_pct_gdp'] > 15:
        return 'Agriculture-significant'
    if row['industry_pct_gdp'] > 35:
        return 'Industry-heavy'
    if row['services_pct_gdp'] > 60:
        return 'Service-dominant'
    return 'Mixed'



def fetch_json(url: str, cache_path: Path | None = None) -> object:
    if cache_path and cache_path.exists():
        with cache_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open('w', encoding='utf-8') as f:
                    json.dump(data, f)
            return data
            
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"Pokušaj {attempt+1}/{MAX_RETRIES} nakon {wait}s ({e})", file=sys.stderr)
                time.sleep(wait)
                
    raise RuntimeError(f"Greška pri dohvaćanju {url}: {last_err}")


def get_wb_meta(cache_dir: Path) -> dict[str, dict]:
    url = f"{WB_API}/country?format=json&per_page=500"
    data = fetch_json(url, cache_dir / "wb_countries.json")
    
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"Neočekivani oblik odgovora s WB /country: {type(data)}")

    out = {}
    for c in data[1]:
        if not isinstance(c, dict):
            continue
            
        region = c.get('region') or {}
        if region.get('id') == 'NA':  # Preskoči agregate
            continue
            
        iso3 = c.get('id')
        if not iso3:
            continue

        try:
            lat = float(c.get('latitude')) if c.get('latitude') else None
        except (TypeError, ValueError):
            lat = None

        out[iso3] = {
            'country_name':  (c.get('name') or '').strip(),
            'capital':       (c.get('capitalCity') or '').strip(),
            'income_group':  ((c.get('incomeLevel') or {}).get('value') or '').strip(),
            'latitude':      lat,
        }
    return out


def get_wb_indicator(code: str, date_range: str, cache_dir: Path) -> dict[str, float]:
    cache_path = cache_dir / f"wb_{code}.json"
    
    if cache_path.exists():
        with cache_path.open('r', encoding='utf-8') as f:
            records = json.load(f)
    else:
        records = []
        page = 1
        while True:
            url = f"{WB_API}/country/all/indicator/{code}?format=json&per_page=2000&date={date_range}&page={page}"
            data = fetch_json(url)
            
            if not isinstance(data, list) or len(data) < 2:
                break
                
            page_records = data[1] if isinstance(data[1], list) else []
            records.extend(page_records)
            
            meta = data[0] if isinstance(data[0], dict) else {}
            try:
                total_pages = int(meta.get('pages', 1))
            except (TypeError, ValueError):
                total_pages = 1
                
            if page >= total_pages or not page_records:
                break
            page += 1
            
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(records, f)

    # Izvlačenje najnovije non-null vrijednosti
    latest = {}
    for r in records:
        if not isinstance(r, dict):
            continue
            
        iso3 = r.get('countryiso3code') or ''
        value = r.get('value')
        date_str = r.get('date')
        
        if not iso3 or value is None or not date_str:
            continue
            
        try:
            year = int(date_str)
            val = float(value)
        except (TypeError, ValueError):
            continue
            
        if iso3 not in latest or year > latest[iso3][0]:
            latest[iso3] = (year, val)
            
    return {iso3: v for iso3, (_, v) in latest.items()}


def get_rc_data(cache_dir: Path) -> dict[str, dict]:
    url = f"{RC_API}?fields=cca3,area,landlocked,borders"
    data = fetch_json(url, cache_dir / "restcountries.json")
    
    if not isinstance(data, list):
        raise RuntimeError(f"Neočekivani oblik odgovora s REST Countries: {type(data)}")

    out = {}
    for c in data:
        if not isinstance(c, dict):
            continue
            
        iso3 = c.get('cca3')
        if not iso3:
            continue
            
        area = c.get('area')
        try:
            area_val = float(area) if area is not None else None
        except (TypeError, ValueError):
            area_val = None
            
        out[iso3] = {
            'area':       area_val,
            'landlocked': bool(c.get('landlocked', False)),
            'borders':    list(c.get('borders') or []),
        }
    return out


def fill_nas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Kaskadno popunjavanje: kontinent+income -> kontinent -> globalni medijan
    for col in cols:
        df[col] = df[col].fillna(df.groupby(['continent', 'income_group'])[col].transform('median'))
        df[col] = df[col].fillna(df.groupby('continent')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median())
    return df


# --- Pipeline ---

def build_dataset(cache_dir: Path, date_range: str) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Dohvaćanje podataka s API-ja i sastavljanje dataseta... (ovo može potrajati)")
    wb_meta = get_wb_meta(cache_dir)

    ind_data = {}
    for code, col in WB_INDICATORS.items():
        ind_data[col] = get_wb_indicator(code, date_range, cache_dir)

    try:
        rc_meta = get_rc_data(cache_dir)
    except Exception as e:
        print(f"UPOZORENJE: Dohvaćanje s REST Countries API-ja nije uspjelo: {e}", file=sys.stderr)
        print("Nastavljam bez toga; podatak o površini oslanjat će se na fallback vrijednosti.", file=sys.stderr)
        rc_meta = {}

    rows = []
    for iso3 in sorted(VALID_COUNTRIES):
        wb_row = wb_meta.get(iso3, {})
        rc_row = rc_meta.get(iso3, {})

        row = {
            'iso3':         iso3,
            'country_name': wb_row.get('country_name') or iso3,
            'capital':      wb_row.get('capital') or CAPITAL_FALLBACKS.get(iso3, ''),
            'income_group': wb_row.get('income_group') or 'Unknown',
        }

        row['region']    = ISO3_TO_SUBREGION.get(iso3)
        row['continent'] = SUBREGION_TO_CONTINENT.get(row['region'])
        row['hemisphere'] = 'Southern' if iso3 in SOUTHERN_HEMISPHERE else 'Northern'
        row['landlocked'] = int(iso3 in LANDLOCKED)
        row['island_nation'] = int(iso3 in ISLAND_NATION)

        area = rc_row.get('area')
        if area and area > 0:
            row['land_area_km2'] = int(round(area))
        else:
            row['land_area_km2'] = LAND_AREA_FALLBACKS.get(iso3)

        row['un_member']           = int(iso3 not in UN_NON_MEMBERS)
        row['eu_member']           = int(iso3 in EU_MEMBERS)
        row['oecd_member']         = int(iso3 in OECD_MEMBERS)
        row['nato_member']         = int(iso3 in NATO_MEMBERS)
        row['g20_member']          = int(iso3 in G20_MEMBERS)
        row['commonwealth_member'] = int(iso3 in COMMONWEALTH_MEMBERS)
        row['opec_member']         = int(iso3 in OPEC_MEMBERS)
        row['primary_language_family'] = LANGUAGE_FAMILY.get(iso3)

        for col in INDICATOR_COLS:
            row[col] = ind_data[col].get(iso3)

        rows.append(row)

    df = pd.DataFrame(rows)

    df['imputed_fields_count'] = df[INDICATOR_COLS].isna().sum(axis=1).astype(int)
    df = fill_nas(df, INDICATOR_COLS)

    for col in ('landlocked', 'island_nation', 'land_area_km2'):
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)

    df['economic_structure'] = df.apply(get_eco_structure, axis=1)
    for col_in, col_out, edges, labels in TIER_SPECS:
        df[col_out] = pd.cut(df[col_in], bins=edges, labels=labels, include_lowest=True).astype(str)

    assert len(df) == 209, f"Očekivano 209 redaka, dobiveno {len(df)}"

    missing_required = [f"{col}={df[col].isna().sum()}" for col in ('continent', 'region', 'primary_language_family') if df[col].isna().sum() > 0]
    if missing_required:
        raise RuntimeError(f"Obavezni izvedeni stupci imaju NaN vrijednosti: {missing_required}")

    extra = set(df.columns) - set(OUTPUT_COLS)
    missing = set(OUTPUT_COLS) - set(df.columns)
    if extra:
        raise RuntimeError(f"Neočekivani stupci: {extra}")
    if missing:
        raise RuntimeError(f"Nedostaju stupci: {missing}")

    print("\nSažetak preuzimanja:")
    print(f"  Države s potpunim WB podacima:  {int((df['imputed_fields_count'] == 0).sum())}/{len(df)}")
    print(f"  Prosječno imputiranih polja:    {float(df['imputed_fields_count'].mean()):.2f}/19")
    print(f"  Države bez podatka o površini:  {int(df['land_area_km2'].isna().sum())}")

    return df[OUTPUT_COLS]


# --- CLI ---

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Izrada countries_complete.csv datoteke s javnih web API-ja.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('output', nargs='?', default='countries_complete.csv', help='putanja do izlazne CSV datoteke')
    ap.add_argument('--cache-dir', default=str(CACHE_DIR), help='direktorij za spremanje HTTP odgovora (cache)')
    ap.add_argument('--date-range', default=DATE_RANGE, help='raspon datuma za WB indikatore (YYYY:YYYY)')
    ap.add_argument('--refresh', action='store_true', help='briše cache prije pokretanja (prisiljava novo preuzimanje)')
    
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    
    if args.refresh and cache_dir.exists():
        print(f"Osvježavanje: brišem cache na {cache_dir}")
        shutil.rmtree(cache_dir)

    df = build_dataset(cache_dir, args.date_range)
    out = Path(args.output)
    df.to_csv(out, index=False)
    print(f"\nZapisano u {out}: {len(df)} redaka x {len(df.columns)} stupaca")


if __name__ == '__main__':
    main()