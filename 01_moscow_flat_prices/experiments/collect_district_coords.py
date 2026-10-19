# Сбор внешних данных: координаты центров районов Москвы из английской Википедии.
# Результат: data/district_coords.csv (District, lat, lon, source). Запускается один раз, ~2 минуты.
# Для 5 районов, у страниц которых нет координат, взяты ориентировочные координаты вручную.
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "district_coords.csv"

WIKI_TITLES = {
    "Академический": "Akademichesky_District",
    "Алексеевский": None,
    "Алтуфьевский": "Altufyevsky_District",
    "Арбат": "Arbat_District",
    "Аэропорт": "Aeroport_District",
    "Бабушкинский": None,
    "Басманный": "Basmanny_District",
    "Беговой": "Begovoy_District",
    "Бескудниковский": "Beskudnikovsky_District",
    "Бибирево": "Bibirevo_District",
    "Бирюлёво Восточное": "Biryulyovo_Vostochnoye_District",
    "Бирюлёво Западное": "Biryulyovo_Zapadnoye_District",
    "Богородский": "Bogorodskoye_District",
    "Братеево": "Brateyevo_District",
    "Бутырский": "Butyrsky_District",
    "Вешняки": "Veshnyaki_District",
    "Внуково": "Vnukovo_District",
    "Войковский": "Voykovsky_District",
    "Восточный": "Vostochny_District,_Moscow",
    "Гагаринский": "Gagarinsky_District,_Moscow",
    "Головинский": "Golovinsky_District",
    "Гольяново": None,
    "Даниловский": "Danilovsky_District,_Moscow",
    "Дегунино Восточное": "Vostochnoye_Degunino_District",
    "Дегунино Западное": "Zapadnoye_Degunino_District",
    "Дмитровский": "Dmitrovsky_District,_Moscow",
    "Донской": "Donskoy_District",
    "Дорогомилово": "Dorogomilovo_District",
    "Жулебино": "Vykhino-Zhulebino_District",
    "Замоскворечье": "Zamoskvorechye_District",
    "Зюзино": "Zyuzino_District",
    "Зябликово": "Zyablikovo_District",
    "Ивановское": "Ivanovskoye_District",
    "Измайлово": "Izmaylovo_District",
    "Измайлово Восточное": "Vostochnoye_Izmaylovo_District",
    "Капотня": "Kapotnya_District",
    "Коньково": "Konkovo_District",
    "Коптево": "Koptevo_District",
    "Косино-Ухтомский": "Kosino-Ukhtomsky_District",
    "Котловка": "Kotlovka_District",
    "Красносельский": "Krasnoselsky_District,_Moscow",
    "Крылатский": "Krylatskoye_District",
    "Кузьминки": "Kuzminki_District",
    "Кунцево": "Kuntsevo_District",
    "Куркино": "Kurkino_District",
    "Левобережный": "Levoberezhny_District,_Moscow",
    "Лефортово": "Lefortovo_District",
    "Лианозово": "Lianozovo_District",
    "Ломоносовский": "Lomonosovsky_District,_Moscow",
    "Лосиноостровский": "Losinoostrovsky_District",
    "Люблино": "Lyublino_District",
    "Марфино": "Marfino_District",
    "Марьина роща": "Maryina_Roshcha_District",
    "Марьино": "Maryino_District",
    "Медведково Южное": "Yuzhnoye_Medvedkovo_District",
    "Метрогородок": "Metrogorodok_District",
    "Мещанский": "Meshchansky_District",
    "Митино": "Mitino_District",
    "Можайский": "Mozhaysky_District,_Moscow",
    "Молжаниновский": "Molzhaninovsky_District",
    "Москворечье-Сабурово": "Moskvorechye-Saburovo_District",
    "Нагатино Садовники": "Nagatino-Sadovniki_District",
    "Нагатинский затон": "Nagatinsky_Zaton_District",
    "Нагорный": "Nagorny_District",
    "Некрасовка": "Nekrasovka_District",
    "Нижегородский": "Nizhegorodsky_District,_Moscow",
    "Ново-Переделкино": "Novo-Peredelkino_District",
    "Новогиреево": "Novogireyevo_District",
    "Новокосино": "Novokosino_District",
    "Обручевский": "Obruchevsky_District",
    "Останкино": "Ostankinsky_District",
    "Отрадное": "Otradnoye_District",
    "Очаково-Матвеевское": "Ochakovo-Matveyevskoye_District",
    "Перово": "Perovo_District",
    "Печатники": "Pechatniki_District",
    "Покровское Стрешнево": "Pokrovskoye-Streshnevo_District",
    "Преображенский": "Preobrazhenskoye_District",
    "Пресненский": None,
    "Район Вернадского": "Prospekt_Vernadskogo_District",
    "Раменки": "Ramenki_District",
    "Ростокино": "Rostokino_District",
    "Рязанский": "Ryazansky_District,_Moscow",
    "Савеловский": "Savyolovsky_District",
    "Свиблово": "Sviblovo_District",
    "Северное Бутово": "Severnoye_Butovo_District",
    "Северное Измайлово": "Severnoye_Izmaylovo_District",
    "Северное Медведково": "Severnoye_Medvedkovo_District",
    "Северное Орехово Борисово": "Orekhovo-Borisovo_Severnoye_District",
    "Северное Тушино": "Severnoye_Tushino_District",
    "Северное Чертаново": "Chertanovo_Severnoye_District",
    "Северный": "Severny_District,_Moscow",
    "Сокол": "Sokol_District",
    "Соколиная гора": "Sokolinaya_Gora_District",
    "Сокольники": "Sokolniki_District",
    "Солнцево": None,
    "Строгино": "Strogino_District",
    "Таганка": "Tagansky_District",
    "Тверской": "Tverskoy_District",
    "Текстильщики": "Tekstilshchiki_District",
    "Тимирязевский": "Timiryazevsky_District",
    "Тропарево Никулино": "Troparyovo-Nikulino_District",
    "Тёплый Стан": "Tyoply_Stan_District",
    "Филевский парк": "Filyovsky_Park_District",
    "Фили Давыдково": "Fili-Davydkovo_District",
    "Хамовники": "Khamovniki_District",
    "Ховрино": "Khovrino_District",
    "Хорошевский": "Khoroshyovsky_District",
    "Хорошёво-Мневники": "Khoroshyovo-Mnyovniki_District",
    "Царицыно": "Tsaritsyno_District",
    "Центральное Чертаново": "Chertanovo_Tsentralnoye_District",
    "Черемушки": "Cheryomushki_District",
    "Щукино": "Shchukino_District",
    "Южное Бутово": "Yuzhnoye_Butovo_District",
    "Южное Орехово Борисово": "Orekhovo-Borisovo_Yuzhnoye_District",
    "Южное Тушино": "Yuzhnoye_Tushino_District",
    "Южное Чертаново": "Chertanovo_Yuzhnoye_District",
    "Южнопортовый": "Yuzhnoportovy_District",
    "Якиманка": "Yakimanka_District",
    "Ярославский": "Yaroslavsky_District,_Moscow",
    "Ясенево": "Yasenevo_District"
}

MANUAL = {  # страницы без координат: центр района по карте
    "Алексеевский": (55.807, 37.640),
    "Бабушкинский": (55.868, 37.664),
    "Гольяново": (55.823, 37.818),
    "Пресненский": (55.758, 37.560),
    "Солнцево": (55.649, 37.395),
}


def fetch_coords(title):
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title)
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-pro-study/1.0"})
    for attempt in range(3):
        try:
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
            c = re.search(r'"wgCoordinates":\{"lat":([\d.]+),"lon":([\d.]+)\}', html)
            return (float(c.group(1)), float(c.group(2))) if c else None
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def collect():
    rows = []
    for district, title in WIKI_TITLES.items():
        if district in MANUAL:
            rows.append((district, *MANUAL[district], "manual"))
            continue
        ll = fetch_coords(title)
        rows.append((district, *(ll or (None, None)), "wikipedia" if ll else "missing"))
        time.sleep(0.7)
    return pd.DataFrame(rows, columns=["District", "lat", "lon", "source"])


if __name__ == "__main__":
    df = collect()
    df.to_csv(OUT, index=False)
    print(df.source.value_counts().to_dict())
