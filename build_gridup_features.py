import csv
import math
import zipfile
from datetime import date, timedelta
from pathlib import Path


OUT_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
MAIN_CSV = OUT_DIR / "gridup_features.csv"
SOURCES_CSV = OUT_DIR / "gridup_sources.csv"
ZIP_PATH = OUT_DIR / "gridup_dataset.zip"

CUTOFF = date(2026, 3, 31)
START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 7, 31)


# ============================================================
# MGM 1991-2020 resmi uzun dönem iklim normalleri
#
# ÖNEMLİ:
# Nisan-Temmuz 2026 gerçekleşmiş hava durumu kullanılmaz.
# 31 Mart 2026 sonrasında oluşmuş meteorolojik gözlem kullanılmaz.
#
# 2026 tahmin dönemindeki hava özellikleri yalnız:
# - 1991-2020 uzun dönem iklim normalleri
# - deterministik astronomik hesaplamalar
# üzerinden oluşturulur.
# ============================================================

CITY_NORMALS = {
    "IZMIR": {
        "lat": 38.4237,
        "lon": 27.1428,
        "tmean": [
            9.0,
            9.9,
            12.4,
            16.2,
            21.1,
            26.0,
            28.6,
            28.5,
            24.2,
            19.5,
            14.4,
            10.5,
        ],
        "tmax": [
            12.7,
            14.0,
            17.2,
            21.3,
            26.5,
            31.3,
            33.8,
            33.6,
            29.5,
            24.6,
            18.8,
            14.0,
        ],
        "tmin": [
            6.0,
            6.6,
            8.6,
            11.8,
            16.2,
            20.9,
            23.5,
            23.7,
            19.5,
            15.4,
            10.9,
            7.7,
        ],
        "sun": [
            4.5,
            5.2,
            6.6,
            7.9,
            9.7,
            11.5,
            12.3,
            11.6,
            9.7,
            7.6,
            5.8,
            4.2,
        ],
        "rain_days": [
            11.57,
            12.00,
            10.23,
            9.00,
            7.10,
            3.67,
            0.67,
            0.83,
            3.07,
            6.67,
            9.07,
            13.30,
        ],
        "precip_month": [
            127.5,
            107.2,
            77.8,
            50.1,
            32.9,
            14.4,
            3.0,
            6.7,
            23.5,
            56.5,
            99.6,
            131.3,
        ],
        "climate_profile": "coastal",
    },

    "MANISA": {
        "lat": 38.6191,
        "lon": 27.4289,
        "tmean": [
            6.3,
            7.9,
            11.0,
            15.2,
            20.7,
            25.7,
            28.6,
            28.5,
            23.7,
            18.2,
            11.9,
            7.8,
        ],
        "tmax": [
            10.7,
            12.9,
            16.8,
            21.7,
            27.6,
            32.7,
            35.7,
            35.7,
            31.1,
            24.8,
            17.6,
            12.0,
        ],
        "tmin": [
            3.0,
            4.1,
            6.1,
            9.6,
            14.4,
            19.1,
            22.2,
            22.3,
            17.5,
            13.1,
            7.7,
            4.7,
        ],
        "sun": [
            2.5,
            3.4,
            4.7,
            5.4,
            7.2,
            8.9,
            9.3,
            9.0,
            7.5,
            5.5,
            3.4,
            1.9,
        ],
        "rain_days": [
            10.80,
            11.00,
            9.77,
            9.03,
            7.07,
            3.77,
            1.20,
            1.00,
            3.03,
            6.13,
            8.30,
            12.20,
        ],
        "precip_month": [
            123.5,
            108.4,
            75.9,
            54.9,
            39.0,
            25.1,
            7.7,
            11.2,
            22.8,
            53.8,
            85.5,
            116.8,
        ],
        "climate_profile": "inland",
    },
}


# ============================================================
# Tatiller
#
# Yalnız cutoff tarihinde önceden bilinebilen takvim olayları.
# ============================================================

HOLIDAYS = {
    # 2025
    date(2025, 1, 1): (
        "new_year",
        1.0,
        "Yılbaşı",
    ),

    date(2025, 3, 29): (
        "ramadan_eve",
        0.5,
        "Ramazan Bayramı Arifesi",
    ),

    date(2025, 3, 30): (
        "ramadan_feast_1",
        1.0,
        "Ramazan Bayramı 1",
    ),

    date(2025, 3, 31): (
        "ramadan_feast_2",
        1.0,
        "Ramazan Bayramı 2",
    ),

    date(2025, 4, 1): (
        "ramadan_feast_3",
        1.0,
        "Ramazan Bayramı 3",
    ),

    date(2025, 4, 23): (
        "national_holiday",
        1.0,
        "23 Nisan",
    ),

    date(2025, 5, 1): (
        "labour_day",
        1.0,
        "1 Mayıs",
    ),

    date(2025, 5, 19): (
        "national_holiday",
        1.0,
        "19 Mayıs",
    ),

    date(2025, 6, 5): (
        "sacrifice_eve",
        0.5,
        "Kurban Bayramı Arifesi",
    ),

    date(2025, 6, 6): (
        "sacrifice_feast_1",
        1.0,
        "Kurban Bayramı 1",
    ),

    date(2025, 6, 7): (
        "sacrifice_feast_2",
        1.0,
        "Kurban Bayramı 2",
    ),

    date(2025, 6, 8): (
        "sacrifice_feast_3",
        1.0,
        "Kurban Bayramı 3",
    ),

    date(2025, 6, 9): (
        "sacrifice_feast_4",
        1.0,
        "Kurban Bayramı 4",
    ),

    date(2025, 7, 15): (
        "national_holiday",
        1.0,
        "15 Temmuz",
    ),

    date(2025, 8, 30): (
        "national_holiday",
        1.0,
        "30 Ağustos",
    ),

    date(2025, 10, 28): (
        "republic_eve",
        0.5,
        "Cumhuriyet Bayramı Arifesi",
    ),

    date(2025, 10, 29): (
        "republic_day",
        1.0,
        "29 Ekim",
    ),

    # 2026
    date(2026, 1, 1): (
        "new_year",
        1.0,
        "Yılbaşı",
    ),

    date(2026, 3, 19): (
        "ramadan_eve",
        0.5,
        "Ramazan Bayramı Arifesi",
    ),

    date(2026, 3, 20): (
        "ramadan_feast_1",
        1.0,
        "Ramazan Bayramı 1",
    ),

    date(2026, 3, 21): (
        "ramadan_feast_2",
        1.0,
        "Ramazan Bayramı 2",
    ),

    date(2026, 3, 22): (
        "ramadan_feast_3",
        1.0,
        "Ramazan Bayramı 3",
    ),

    date(2026, 4, 23): (
        "national_holiday",
        1.0,
        "23 Nisan",
    ),

    date(2026, 5, 1): (
        "labour_day",
        1.0,
        "1 Mayıs",
    ),

    date(2026, 5, 19): (
        "national_holiday",
        1.0,
        "19 Mayıs",
    ),

    date(2026, 5, 26): (
        "sacrifice_eve",
        0.5,
        "Kurban Bayramı Arifesi",
    ),

    date(2026, 5, 27): (
        "sacrifice_feast_1",
        1.0,
        "Kurban Bayramı 1",
    ),

    date(2026, 5, 28): (
        "sacrifice_feast_2",
        1.0,
        "Kurban Bayramı 2",
    ),

    date(2026, 5, 29): (
        "sacrifice_feast_3",
        1.0,
        "Kurban Bayramı 3",
    ),

    date(2026, 5, 30): (
        "sacrifice_feast_4",
        1.0,
        "Kurban Bayramı 4",
    ),

    date(2026, 7, 15): (
        "national_holiday",
        1.0,
        "15 Temmuz",
    ),
}


RAMADAN_RANGES = {
    2025: (
        date(2025, 3, 1),
        date(2025, 3, 29),
    ),

    2026: (
        date(2026, 2, 19),
        date(2026, 3, 19),
    ),
}


KURBAN_RANGES = {
    2025: (
        date(2025, 6, 6),
        date(2025, 6, 9),
    ),

    2026: (
        date(2026, 5, 27),
        date(2026, 5, 30),
    ),
}


def school_state(d):
    """
    MEB tarafından cutoff'tan önce yayımlanmış
    eğitim-öğretim takvimlerinden okul durumu.
    """

    if date(2025, 1, 1) <= d <= date(2025, 1, 17):
        return "in_session"

    if date(2025, 1, 18) <= d <= date(2025, 2, 2):
        return "semester_break"

    if date(2025, 2, 3) <= d <= date(2025, 3, 30):
        return "in_session"

    if date(2025, 3, 31) <= d <= date(2025, 4, 6):
        return "midterm_break"

    if date(2025, 4, 7) <= d <= date(2025, 6, 20):
        return "in_session"

    if date(2025, 6, 21) <= d <= date(2025, 9, 7):
        return "summer_break"

    if date(2025, 9, 8) <= d <= date(2025, 11, 9):
        return "in_session"

    if date(2025, 11, 10) <= d <= date(2025, 11, 16):
        return "midterm_break"

    if date(2025, 11, 17) <= d <= date(2026, 1, 18):
        return "in_session"

    if date(2026, 1, 19) <= d <= date(2026, 2, 1):
        return "semester_break"

    if date(2026, 2, 2) <= d <= date(2026, 3, 15):
        return "in_session"

    if date(2026, 3, 16) <= d <= date(2026, 3, 22):
        return "midterm_break"

    if date(2026, 3, 23) <= d <= date(2026, 6, 26):
        return "in_session"

    if date(2026, 6, 27) <= d <= END_DATE:
        return "summer_break"

    return "unknown"


def days_in_month(year, month):
    if month == 12:
        return (
            date(year + 1, 1, 1)
            - date(year, 12, 1)
        ).days

    return (
        date(year, month + 1, 1)
        - date(year, month, 1)
    ).days


def monthly_midpoint_doy(year, month):
    """
    Aylık MGM normallerini günlük değerlere
    yumuşak şekilde dönüştürmek için
    ay orta noktası.
    """

    return (
        date(year, month, 1).timetuple().tm_yday
        + (days_in_month(year, month) - 1) / 2.0
    )


def cyclic_month_interp(d, values):
    """
    12 aylık iklim normalini günlük
    yumuşak climatology serisine dönüştürür.

    Bu interpolasyon gerçek 2026 hava durumu değildir.
    """

    year = d.year
    doy = d.timetuple().tm_yday

    mids = [
        monthly_midpoint_doy(year, month)
        for month in range(1, 13)
    ]

    year_len = (
        366
        if date(year, 12, 31).timetuple().tm_yday == 366
        else 365
    )

    points = (
        [(mids[-1] - year_len, values[-1])]
        + list(zip(mids, values))
        + [(mids[0] + year_len, values[0])]
    )

    for (x0, v0), (x1, v1) in zip(
        points[:-1],
        points[1:],
    ):
        if x0 <= doy <= x1:
            if x1 == x0:
                return v0

            ratio = (
                (doy - x0)
                / (x1 - x0)
            )

            return (
                v0
                + ratio * (v1 - v0)
            )

    return values[d.month - 1]


def solar_geometry(lat_deg, doy):
    """
    FAO-56 astronomik güneş geometrisi.

    Girdi:
    - enlem
    - yılın günü

    Çıktılar gerçek hava ölçümü değildir.
    """

    phi = math.radians(lat_deg)

    dr = (
        1
        + 0.033
        * math.cos(
            2
            * math.pi
            * doy
            / 365.0
        )
    )

    delta = (
        0.409
        * math.sin(
            2
            * math.pi
            * doy
            / 365.0
            - 1.39
        )
    )

    x = (
        -math.tan(phi)
        * math.tan(delta)
    )

    x = max(
        -1.0,
        min(1.0, x),
    )

    sunset_hour_angle = math.acos(x)

    solar_constant = 0.0820

    extraterrestrial_radiation = (
        (24 * 60 / math.pi)
        * solar_constant
        * dr
        * (
            sunset_hour_angle
            * math.sin(phi)
            * math.sin(delta)
            + math.cos(phi)
            * math.cos(delta)
            * math.sin(
                sunset_hour_angle
            )
        )
    )

    daylight_hours = (
        24
        / math.pi
        * sunset_hour_angle
    )

    return (
        delta,
        sunset_hour_angle,
        extraterrestrial_radiation,
        daylight_hours,
    )


EVENT_DATES = sorted(HOLIDAYS.keys())


def nearest_event_features(d):
    previous_events = [
        x
        for x in EVENT_DATES
        if x <= d
    ]

    next_events = [
        x
        for x in EVENT_DATES
        if x >= d
    ]

    prev_date = (
        previous_events[-1]
        if previous_events
        else None
    )

    next_date = (
        next_events[0]
        if next_events
        else None
    )

    days_since_prev = (
        (d - prev_date).days
        if prev_date
        else 999
    )

    days_to_next = (
        (next_date - d).days
        if next_date
        else 999
    )

    if abs(days_since_prev) <= abs(days_to_next):
        signed = days_since_prev
    else:
        signed = -days_to_next

    return (
        days_since_prev,
        days_to_next,
        signed,
        abs(signed),
    )


def bridge_candidate(d):
    """
    İki tatil/off-day arasında kalan çalışma günü.
    """

    if (
        d.weekday() >= 5
        or d in HOLIDAYS
    ):
        return 0

    previous_day = (
        d
        - timedelta(days=1)
    )

    next_day = (
        d
        + timedelta(days=1)
    )

    previous_off = (
        previous_day.weekday() >= 5
        or (
            previous_day in HOLIDAYS
            and HOLIDAYS[previous_day][1] >= 0.5
        )
    )

    next_off = (
        next_day.weekday() >= 5
        or (
            next_day in HOLIDAYS
            and HOLIDAYS[next_day][1] >= 0.5
        )
    )

    return int(
        previous_off
        and next_off
    )


def ramadan_info(d):
    period = RAMADAN_RANGES.get(
        d.year
    )

    if not period:
        return 0, 0

    start, end = period

    if start <= d <= end:
        return (
            1,
            (d - start).days + 1,
        )

    return 0, 0


def kurban_info(d):
    period = KURBAN_RANGES.get(
        d.year
    )

    if not period:
        return 0, 0

    start, end = period

    if start <= d <= end:
        return (
            1,
            (d - start).days + 1,
        )

    return 0, 0


def season_name(month):
    if month in (
        12,
        1,
        2,
    ):
        return "winter"

    if month in (
        3,
        4,
        5,
    ):
        return "spring"

    if month in (
        6,
        7,
        8,
    ):
        return "summer"

    return "autumn"


HEADERS = [
    "date",
    "il",
    "lat",
    "lon",
    "climate_profile",

    "information_cutoff",
    "available_by_cutoff",
    "post_cutoff_realized_weather_used",
    "weather_basis",
    "is_prediction_horizon",

    "year",
    "month",
    "quarter",
    "day",
    "day_of_year",
    "week_of_year",
    "day_of_week",

    "is_weekend",
    "is_monday",
    "is_friday",
    "is_month_start",
    "is_month_end",
    "season",

    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",

    "holiday_type",
    "holiday_name",
    "is_public_holiday",
    "is_half_day_holiday",
    "holiday_day_fraction",

    "days_since_prev_holiday",
    "days_to_next_holiday",
    "event_distance_signed",
    "event_distance_abs",

    "is_pre_holiday_1d",
    "is_post_holiday_1d",

    "is_pre_holiday_3d",
    "is_post_holiday_3d",

    "is_pre_holiday_7d",
    "is_post_holiday_7d",

    "is_bridge_candidate",

    "is_ramadan",
    "ramadan_day_number",
    "is_ramadan_eve",
    "is_ramadan_feast",

    "is_sacrifice_eve",
    "is_sacrifice_feast",
    "sacrifice_feast_day_number",

    "school_state",
    "school_vacation",
    "is_school_day",
    "school_day_fraction",

    "base_workday_fraction",

    "clim_tmean_c",
    "clim_tmax_c",
    "clim_tmin_c",
    "clim_temp_range_c",

    "clim_sunshine_hours",

    "clim_rainy_days_month",
    "clim_precip_month_mm",
    "clim_precip_mm_day",
    "clim_rain_probability",

    "cdd18",
    "cdd22",
    "hdd15",
    "hdd18",

    "solar_declination_rad",
    "sunset_hour_angle_rad",
    "daylight_hours",

    "extraterrestrial_radiation_mj_m2",

    "sunshine_fraction",

    "estimated_solar_radiation_mj_m2",
    "estimated_solar_radiation_kwh_m2",

    "clear_sky_radiation_mj_m2",
    "solar_cloudiness_proxy",

    "hargreaves_et0_mm",
    "clim_water_deficit_mm",
    "irrigation_stress_mm",

    "cooling_solar_interaction",
    "heating_solar_interaction",

    "cdd18_roll7",
    "cdd18_roll30",

    "hdd15_roll7",
    "hdd15_roll30",

    "irrigation_stress_roll7",
    "irrigation_stress_roll30",

    "source_bundle_id",
]


def build():
    rows = []

    for il, meta in CITY_NORMALS.items():

        city_rows = []

        d = START_DATE

        while d <= END_DATE:

            doy = (
                d
                .timetuple()
                .tm_yday
            )

            # ----------------------------------------
            # Climatology
            # ----------------------------------------

            tmean = cyclic_month_interp(
                d,
                meta["tmean"],
            )

            tmax = cyclic_month_interp(
                d,
                meta["tmax"],
            )

            tmin = cyclic_month_interp(
                d,
                meta["tmin"],
            )

            sunshine = cyclic_month_interp(
                d,
                meta["sun"],
            )

            rain_days = (
                meta["rain_days"][
                    d.month - 1
                ]
            )

            precip_month = (
                meta["precip_month"][
                    d.month - 1
                ]
            )

            month_days = days_in_month(
                d.year,
                d.month,
            )

            precip_day = (
                precip_month
                / month_days
            )

            rain_probability = min(
                1.0,
                rain_days
                / month_days,
            )

            # ----------------------------------------
            # Solar geometry
            # ----------------------------------------

            (
                solar_declination,
                sunset_hour_angle,
                ra,
                daylight,
            ) = solar_geometry(
                meta["lat"],
                doy,
            )

            sunshine = min(
                max(
                    0.0,
                    sunshine,
                ),
                daylight,
            )

            sunshine_fraction = (
                sunshine / daylight
                if daylight > 0
                else 0.0
            )

            # Angstrom-Prescott approximation
            estimated_solar_radiation = (
                (
                    0.25
                    + 0.50
                    * sunshine_fraction
                )
                * ra
            )

            clear_sky_radiation = (
                0.75
                * ra
            )

            estimated_solar_kwh = (
                estimated_solar_radiation
                / 3.6
            )

            if clear_sky_radiation > 0:
                cloudiness_proxy = (
                    1.0
                    - min(
                        1.0,
                        estimated_solar_radiation
                        / clear_sky_radiation,
                    )
                )
            else:
                cloudiness_proxy = 0.0

            # ----------------------------------------
            # Temperature / degree days
            # ----------------------------------------

            temp_range = max(
                0.0,
                tmax - tmin,
            )

            cdd18 = max(
                0.0,
                tmean - 18.0,
            )

            cdd22 = max(
                0.0,
                tmean - 22.0,
            )

            hdd15 = max(
                0.0,
                15.0 - tmean,
            )

            hdd18 = max(
                0.0,
                18.0 - tmean,
            )

            # ----------------------------------------
            # Hargreaves ET0 proxy
            # ----------------------------------------

            et0 = max(
                0.0,
                0.0023
                * (tmean + 17.8)
                * math.sqrt(
                    temp_range
                )
                * ra,
            )

            water_deficit = (
                et0
                - precip_day
            )

            irrigation_stress = max(
                0.0,
                water_deficit,
            )

            # ----------------------------------------
            # Calendar
            # ----------------------------------------

            (
                holiday_type,
                holiday_fraction,
                holiday_name,
            ) = HOLIDAYS.get(
                d,
                (
                    "regular",
                    0.0,
                    "",
                ),
            )

            is_public_holiday = int(
                holiday_fraction
                >= 1.0
            )

            is_half_day_holiday = int(
                holiday_fraction
                == 0.5
            )

            (
                days_since_prev,
                days_to_next,
                event_signed,
                event_abs,
            ) = nearest_event_features(
                d
            )

            (
                is_ramadan,
                ramadan_day,
            ) = ramadan_info(
                d
            )

            (
                is_kurban,
                kurban_day,
            ) = kurban_info(
                d
            )

            # ----------------------------------------
            # School calendar
            # ----------------------------------------

            state = school_state(
                d
            )

            school_vacation = int(
                state
                != "in_session"
            )

            school_day_fraction = 0.0

            if (
                state == "in_session"
                and d.weekday() < 5
            ):
                school_day_fraction = 1.0

                if is_public_holiday:
                    school_day_fraction = 0.0

                elif is_half_day_holiday:
                    school_day_fraction = 0.5

            is_school_day = int(
                school_day_fraction > 0
            )

            # ----------------------------------------
            # Work-day proxy
            # ----------------------------------------

            if (
                d.weekday() >= 5
                or is_public_holiday
            ):
                base_workday_fraction = 0.0

            elif is_half_day_holiday:
                base_workday_fraction = 0.5

            else:
                base_workday_fraction = 1.0

            # ----------------------------------------
            # Row
            # ----------------------------------------

            row = {
                "date": d.isoformat(),

                "il": il,

                "lat": meta["lat"],

                "lon": meta["lon"],

                "climate_profile": (
                    meta[
                        "climate_profile"
                    ]
                ),

                "information_cutoff": (
                    CUTOFF.isoformat()
                ),

                "available_by_cutoff": 1,

                "post_cutoff_realized_weather_used": 0,

                "weather_basis": (
                    "MGM_1991_2020_normals_"
                    "plus_deterministic_"
                    "solar_geometry"
                ),

                "is_prediction_horizon": int(
                    d > CUTOFF
                ),

                "year": d.year,

                "month": d.month,

                "quarter": (
                    (d.month - 1)
                    // 3
                    + 1
                ),

                "day": d.day,

                "day_of_year": doy,

                "week_of_year": (
                    d
                    .isocalendar()
                    .week
                ),

                "day_of_week": (
                    d.weekday()
                ),

                "is_weekend": int(
                    d.weekday()
                    >= 5
                ),

                "is_monday": int(
                    d.weekday()
                    == 0
                ),

                "is_friday": int(
                    d.weekday()
                    == 4
                ),

                "is_month_start": int(
                    d.day
                    == 1
                ),

                "is_month_end": int(
                    (
                        d
                        + timedelta(days=1)
                    ).month
                    != d.month
                ),

                "season": season_name(
                    d.month
                ),

                "dow_sin": math.sin(
                    2
                    * math.pi
                    * d.weekday()
                    / 7.0
                ),

                "dow_cos": math.cos(
                    2
                    * math.pi
                    * d.weekday()
                    / 7.0
                ),

                "doy_sin": math.sin(
                    2
                    * math.pi
                    * doy
                    / 365.2425
                ),

                "doy_cos": math.cos(
                    2
                    * math.pi
                    * doy
                    / 365.2425
                ),

                "month_sin": math.sin(
                    2
                    * math.pi
                    * (d.month - 1)
                    / 12.0
                ),

                "month_cos": math.cos(
                    2
                    * math.pi
                    * (d.month - 1)
                    / 12.0
                ),

                "holiday_type": (
                    holiday_type
                ),

                "holiday_name": (
                    holiday_name
                ),

                "is_public_holiday": (
                    is_public_holiday
                ),

                "is_half_day_holiday": (
                    is_half_day_holiday
                ),

                "holiday_day_fraction": (
                    holiday_fraction
                ),

                "days_since_prev_holiday": (
                    days_since_prev
                ),

                "days_to_next_holiday": (
                    days_to_next
                ),

                "event_distance_signed": (
                    event_signed
                ),

                "event_distance_abs": (
                    event_abs
                ),

                "is_pre_holiday_1d": int(
                    days_to_next == 1
                ),

                "is_post_holiday_1d": int(
                    days_since_prev == 1
                ),

                "is_pre_holiday_3d": int(
                    0
                    < days_to_next
                    <= 3
                ),

                "is_post_holiday_3d": int(
                    0
                    < days_since_prev
                    <= 3
                ),

                "is_pre_holiday_7d": int(
                    0
                    < days_to_next
                    <= 7
                ),

                "is_post_holiday_7d": int(
                    0
                    < days_since_prev
                    <= 7
                ),

                "is_bridge_candidate": (
                    bridge_candidate(
                        d
                    )
                ),

                "is_ramadan": (
                    is_ramadan
                ),

                "ramadan_day_number": (
                    ramadan_day
                ),

                "is_ramadan_eve": int(
                    holiday_type
                    == "ramadan_eve"
                ),

                "is_ramadan_feast": int(
                    holiday_type.startswith(
                        "ramadan_feast"
                    )
                ),

                "is_sacrifice_eve": int(
                    holiday_type
                    == "sacrifice_eve"
                ),

                "is_sacrifice_feast": (
                    is_kurban
                ),

                "sacrifice_feast_day_number": (
                    kurban_day
                ),

                "school_state": (
                    state
                ),

                "school_vacation": (
                    school_vacation
                ),

                "is_school_day": (
                    is_school_day
                ),

                "school_day_fraction": (
                    school_day_fraction
                ),

                "base_workday_fraction": (
                    base_workday_fraction
                ),

                "clim_tmean_c": (
                    tmean
                ),

                "clim_tmax_c": (
                    tmax
                ),

                "clim_tmin_c": (
                    tmin
                ),

                "clim_temp_range_c": (
                    temp_range
                ),

                "clim_sunshine_hours": (
                    sunshine
                ),

                "clim_rainy_days_month": (
                    rain_days
                ),

                "clim_precip_month_mm": (
                    precip_month
                ),

                "clim_precip_mm_day": (
                    precip_day
                ),

                "clim_rain_probability": (
                    rain_probability
                ),

                "cdd18": (
                    cdd18
                ),

                "cdd22": (
                    cdd22
                ),

                "hdd15": (
                    hdd15
                ),

                "hdd18": (
                    hdd18
                ),

                "solar_declination_rad": (
                    solar_declination
                ),

                "sunset_hour_angle_rad": (
                    sunset_hour_angle
                ),

                "daylight_hours": (
                    daylight
                ),

                "extraterrestrial_radiation_mj_m2": (
                    ra
                ),

                "sunshine_fraction": (
                    sunshine_fraction
                ),

                "estimated_solar_radiation_mj_m2": (
                    estimated_solar_radiation
                ),

                "estimated_solar_radiation_kwh_m2": (
                    estimated_solar_kwh
                ),

                "clear_sky_radiation_mj_m2": (
                    clear_sky_radiation
                ),

                "solar_cloudiness_proxy": (
                    cloudiness_proxy
                ),

                "hargreaves_et0_mm": (
                    et0
                ),

                "clim_water_deficit_mm": (
                    water_deficit
                ),

                "irrigation_stress_mm": (
                    irrigation_stress
                ),

                "cooling_solar_interaction": (
                    cdd18
                    * estimated_solar_kwh
                ),

                "heating_solar_interaction": (
                    hdd15
                    * estimated_solar_kwh
                ),

                "source_bundle_id": (
                    "GRIDUP_SAFE_V1"
                ),
            }

            city_rows.append(
                row
            )

            d += timedelta(
                days=1
            )

        # ====================================================
        # Rolling climatology
        #
        # Burada da gerçekleşmiş gelecek hava yok.
        # Yalnız climatology-derived değerlerin rolling
        # ortalamaları var.
        # ====================================================

        for i, row in enumerate(
            city_rows
        ):

            rolling_specs = [
                (
                    "cdd18",
                    "cdd18_roll7",
                    "cdd18_roll30",
                ),

                (
                    "hdd15",
                    "hdd15_roll7",
                    "hdd15_roll30",
                ),

                (
                    "irrigation_stress_mm",
                    "irrigation_stress_roll7",
                    "irrigation_stress_roll30",
                ),
            ]

            for (
                base,
                out7,
                out30,
            ) in rolling_specs:

                vals7 = [
                    city_rows[j][base]
                    for j in range(
                        max(
                            0,
                            i - 6,
                        ),
                        i + 1,
                    )
                ]

                vals30 = [
                    city_rows[j][base]
                    for j in range(
                        max(
                            0,
                            i - 29,
                        ),
                        i + 1,
                    )
                ]

                row[out7] = (
                    sum(vals7)
                    / len(vals7)
                )

                row[out30] = (
                    sum(vals30)
                    / len(vals30)
                )

        rows.extend(
            city_rows
        )

    # ========================================================
    # Deterministic ordering
    # ========================================================

    rows.sort(
        key=lambda r: (
            r["date"],
            r["il"],
        )
    )

    # Compact / reproducible float output.
    for row in rows:
        for key, value in list(
            row.items()
        ):
            if isinstance(
                value,
                float,
            ):
                row[key] = round(
                    value,
                    6,
                )

    # ========================================================
    # Main feature CSV
    # ========================================================

    with MAIN_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=HEADERS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Source manifest
    # ========================================================

    sources = [
        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "MGM İzmir 1991-2020 "
                "resmi iklim normalleri"
            ),

            "url": (
                "https://www.mgm.gov.tr/"
                "veridegerlendirme/"
                "il-ve-ilceler-istatistik.aspx"
                "?k=H&m=IZMIR"
            ),

            "used_for": (
                "Aylık sıcaklık, min/max sıcaklık, "
                "güneşlenme, yağışlı gün ve "
                "yağış normalleri"
            ),

            "cutoff_safety": (
                "1991-2020 uzun dönem normalleri; "
                "31 Mart 2026 öncesinde yayımlanmış; "
                "gelecekte gerçekleşmiş hava yok"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "MGM Manisa 1991-2020 "
                "resmi iklim normalleri"
            ),

            "url": (
                "https://www.mgm.gov.tr/"
                "Veridegerlendirme/"
                "il-ve-ilceler-istatistik.aspx"
                "?k=H&m=MANISA"
            ),

            "used_for": (
                "Aylık sıcaklık, min/max sıcaklık, "
                "güneşlenme, yağışlı gün ve "
                "yağış normalleri"
            ),

            "cutoff_safety": (
                "1991-2020 uzun dönem normalleri; "
                "31 Mart 2026 öncesinde yayımlanmış; "
                "gelecekte gerçekleşmiş hava yok"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "Diyanet 2025 dini günler "
                "ve resmi tatiller"
            ),

            "url": (
                "https://vakithesaplama."
                "diyanet.gov.tr/"
                "icerik.php?icerik=152"
            ),

            "used_for": (
                "2025 Ramazan/Kurban tarihleri "
                "ve göreli gün özellikleri"
            ),

            "cutoff_safety": (
                "2025 takvimi; tahmin "
                "cutoff'undan önce bilinen "
                "takvim olguları"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "Diyanet 2026 dini günler"
            ),

            "url": (
                "https://vakithesaplama."
                "diyanet.gov.tr/"
                "dinigunler.php?yil=2026"
            ),

            "used_for": (
                "2026 Ramazan/Kurban tarihleri "
                "ve göreli gün özellikleri"
            ),

            "cutoff_safety": (
                "2026 dini takvim; "
                "31 Mart 2026 itibarıyla "
                "bilinen takvim olguları"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "Diyanet 2026 "
                "resmi tatil günleri"
            ),

            "url": (
                "https://vakithesaplama."
                "diyanet.gov.tr/"
                "icerik.php?icerik=158"
            ),

            "used_for": (
                "2026 resmi tatil / "
                "yarım gün etiketleri"
            ),

            "cutoff_safety": (
                "Kanuni tatil takvimi; "
                "cutoff öncesinde bilinen "
                "deterministik tarih bilgisi"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "MEB 2024-2025 "
                "eğitim öğretim takvimi"
            ),

            "url": (
                "https://www.meb.gov.tr/"
                "2024-2025-egitim-ogretim-yili-"
                "takvimi-aciklandi/"
                "haber/33888/tr"
            ),

            "used_for": (
                "2025 yarıyıl, ara tatil "
                "ve yaz tatili etiketleri"
            ),

            "cutoff_safety": (
                "28 Mayıs 2024'te yayımlandı"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "MEB 2025-2026 "
                "eğitim öğretim takvimi"
            ),

            "url": (
                "https://meb.gov.tr/"
                "2025-2026-egitim-ogretim-yili-"
                "takvimi-aciklandi/"
                "haber/37198/ar"
            ),

            "used_for": (
                "2025-2026 okul açık/kapalı "
                "ve tatil etiketleri"
            ),

            "cutoff_safety": (
                "15 Mayıs 2025'te yayımlandı; "
                "cutoff öncesinde biliniyor"
            ),
        },

        {
            "source_bundle_id": (
                "GRIDUP_SAFE_V1"
            ),

            "source_name": (
                "FAO-56 meteorolojik veri / "
                "radyasyon ve Hargreaves yöntemleri"
            ),

            "url": (
                "https://www.fao.org/"
                "4/X0490E/x0490e07.htm"
            ),

            "used_for": (
                "Gün uzunluğu, extraterrestrial "
                "radiation, Angstrom-Prescott Rs "
                "ve Hargreaves ET0 türevleri"
            ),

            "cutoff_safety": (
                "Deterministik formüller ve "
                "uzun dönem normallerinden "
                "türetilmiştir; post-cutoff "
                "gözlem yok"
            ),
        },
    ]

    with SOURCES_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                sources[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            sources
        )

    # ========================================================
    # ZIP
    # ========================================================

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        z.write(
            MAIN_CSV,
            arcname=MAIN_CSV.name,
        )

        z.write(
            SOURCES_CSV,
            arcname=SOURCES_CSV.name,
        )

    print()
    print("=" * 60)
    print("GRID UP cutoff-safe dataset oluşturuldu.")
    print("=" * 60)

    print(
        f"Rows     : {len(rows):,}"
    )

    print(
        f"Features : {len(HEADERS)}"
    )

    print(
        f"Cutoff   : {CUTOFF}"
    )

    print(
        "Post-cutoff realized weather used: NO"
    )

    print()

    print(
        f"CSV      : {MAIN_CSV.resolve()}"
    )

    print(
        f"Sources  : {SOURCES_CSV.resolve()}"
    )

    print(
        f"ZIP      : {ZIP_PATH.resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    build()
