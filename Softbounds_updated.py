import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go

# ==========================================
# Page Config
# ==========================================
st.set_page_config(page_title="Dairy Assessment Plausibility Check", layout="wide")

st.title("🧭 Dairy Assessment Plausibility Check")
st.caption("Upload a CSV of assessments. Only breeds **fleckvieh, braunvieh, schwarzbunt** are analyzed.")

# ---------------------------
# Constants & Helpers
# ---------------------------

ALLOWED_BREEDS = {"fleckvieh", "braunvieh", "schwarzbunt"}

# Softbounds table loaded from Master Taxonomy CSV. Product fixed to 'Lf L' and Year to 2024.
# Now includes herd_group column (any, small, medium, large)
_softbounds_rows = [
    # Generated from Master Taxonomy Milk LfL - Sheet37.csv
    ("energy_consumption_per_cow","any","Lf L",2024,"any","any",200,1200),
    ("fat_content","braunvieh","Lf L",2024,"Conventional","any",0.0370,0.0490),
    ("fat_content","fleckvieh","Lf L",2024,"Conventional","any",0.0370,0.0490),
    ("fat_content","schwarzbunt","Lf L",2024,"Conventional","any",0.0370,0.0490),
    ("first_calving_age","braunvieh","Lf L",2024,"Conventional","any",25,37),
    ("first_calving_age","fleckvieh","Lf L",2024,"Conventional","any",24,36),
    ("first_calving_age","schwarzbunt","Lf L",2024,"Conventional","any",21,33),
    ("gutting_rate","fleckvieh","Lf L",2024,"Conventional","any",0.4000,0.6220),
    ("gutting_rate","schwarzbunt","Lf L",2024,"Conventional","any",0.3750,0.5900),
    ("gutting_rate","braunvieh","Lf L",2024,"Conventional","any",0.3720,0.5760),
    ("live_weight","schwarzbunt","Lf L",2024,"Conventional","any",600,712),
    ("protein_content","braunvieh","Lf L",2024,"Conventional","any",0.0328,0.0383),
    ("protein_content","fleckvieh","Lf L",2024,"Conventional","any",0.0328,0.0383),
    ("protein_content","schwarzbunt","Lf L",2024,"Conventional","any",0.0320,0.0370),
    ("slaughter_weight","fleckvieh","Lf L",2024,"Conventional","any",299,404),
    ("slaughter_weight","schwarzbunt","Lf L",2024,"Conventional","any",250,365),
    ("slaughter_weight","braunvieh","Lf L",2024,"Conventional","any",249,362),
    ("weaning_age","braunvieh","Lf L",2024,"Conventional","any",4,16),
    ("weaning_age","fleckvieh","Lf L",2024,"Conventional","any",4,16),
    ("weaning_age","schwarzbunt","Lf L",2024,"Conventional","any",4,16),
    ("attrition_rate","fleckvieh","Lf L",2024,"Conventional","large",0.0900,0.4550),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Conventional","large",0.1000,0.4500),
    ("attrition_rate","braunvieh","Lf L",2024,"Conventional","large",0.1000,0.3780),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Conventional","large",0.0050,0.1700),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Conventional","large",0.0150,0.1550),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Conventional","large",0.0020,0.1230),
    ("intercalving_period","braunvieh","Lf L",2024,"Conventional","large",365,460),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Conventional","large",360,460),
    ("intercalving_period","fleckvieh","Lf L",2024,"Conventional","large",350,450),
    ("live_weight","fleckvieh","Lf L",2024,"Conventional","large",650,800),
    ("live_weight","braunvieh","Lf L",2024,"Conventional","large",600,790),
    ("loss_rate","schwarzbunt","Lf L",2024,"Conventional","large",0.0040,0.0990),
    ("loss_rate","braunvieh","Lf L",2024,"Conventional","large",0.0020,0.0780),
    ("loss_rate","fleckvieh","Lf L",2024,"Conventional","large",0.0000,0.0550),
    ("milk_yield","schwarzbunt","Lf L",2024,"Conventional","large",6902,13637),
    ("milk_yield","fleckvieh","Lf L",2024,"Conventional","large",5825,11582),
    ("milk_yield","braunvieh","Lf L",2024,"Conventional","large",5852,11158),
    ("attrition_rate","fleckvieh","Lf L",2024,"Conventional","medium",0.0240,0.4680),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Conventional","medium",0.0450,0.4370),
    ("attrition_rate","braunvieh","Lf L",2024,"Conventional","medium",0.0610,0.3780),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Conventional","medium",0.0000,0.1770),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Conventional","medium",0.0000,0.1700),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Conventional","medium",0.0000,0.1230),
    ("intercalving_period","braunvieh","Lf L",2024,"Conventional","medium",365,480),
    ("intercalving_period","fleckvieh","Lf L",2024,"Conventional","medium",350,470),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Conventional","medium",360,464),
    ("live_weight","fleckvieh","Lf L",2024,"Conventional","medium",650,800),
    ("live_weight","braunvieh","Lf L",2024,"Conventional","medium",600,730),
    ("loss_rate","schwarzbunt","Lf L",2024,"Conventional","medium",0.0000,0.0900),
    ("loss_rate","braunvieh","Lf L",2024,"Conventional","medium",0.0000,0.0800),
    ("loss_rate","fleckvieh","Lf L",2024,"Conventional","medium",0.0000,0.0630),
    ("milk_yield","schwarzbunt","Lf L",2024,"Conventional","medium",4905,12750),
    ("milk_yield","fleckvieh","Lf L",2024,"Conventional","medium",4750,10750),
    ("milk_yield","braunvieh","Lf L",2024,"Conventional","medium",4724,10316),
    ("attrition_rate","fleckvieh","Lf L",2024,"Conventional","small",0.0000,0.4570),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Conventional","small",0.0450,0.4200),
    ("attrition_rate","braunvieh","Lf L",2024,"Conventional","small",0.0000,0.4080),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Conventional","small",0.0000,0.1700),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Conventional","small",0.0000,0.1390),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Conventional","small",0.0000,0.1020),
    ("intercalving_period","braunvieh","Lf L",2024,"Conventional","small",365,500),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Conventional","small",360,495),
    ("intercalving_period","fleckvieh","Lf L",2024,"Conventional","small",350,470),
    ("live_weight","fleckvieh","Lf L",2024,"Conventional","small",620,800),
    ("live_weight","braunvieh","Lf L",2024,"Conventional","small",600,726),
    ("loss_rate","schwarzbunt","Lf L",2024,"Conventional","small",0.0000,0.0870),
    ("loss_rate","braunvieh","Lf L",2024,"Conventional","small",0.0000,0.0810),
    ("loss_rate","fleckvieh","Lf L",2024,"Conventional","small",0.0000,0.0550),
    ("milk_yield","schwarzbunt","Lf L",2024,"Conventional","small",3852,11400),
    ("milk_yield","braunvieh","Lf L",2024,"Conventional","small",3678,10193),
    ("milk_yield","fleckvieh","Lf L",2024,"Conventional","small",2600,10100),
    ("attrition_rate","gelbvieh","Lf L",2024,"Conventional","large",0.150,0.420),
    ("attrition_rate","gelbvieh","Lf L",2024,"Conventional","medium",0.000,0.420),
    ("attrition_rate","gelbvieh","Lf L",2024,"Conventional","small",0.000,0.420),
    ("loss_rate","gelbvieh","Lf L",2024,"Conventional","large",0.005,0.055),
    ("loss_rate","gelbvieh","Lf L",2024,"Conventional","medium",0.000,0.055),
    ("loss_rate","gelbvieh","Lf L",2024,"Conventional","small",0.000,0.055),
    ("intercalving_period","gelbvieh","Lf L",2024,"Conventional","any",379.000,439.000),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Conventional","small",0.000,0.150),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Conventional","medium",0.000,0.150),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Conventional","large",0.023,0.150),
    ("weaning_age","gelbvieh","Lf L",2024,"Conventional","any",4.000,12.000),
    ("live_weight","gelbvieh","Lf L",2024,"Conventional","any",650.000,800.000),
    ("slaughter_weight","gelbvieh","Lf L",2024,"Conventional","any",280.000,369.000),
    ("gutting_rate","gelbvieh","Lf L",2024,"Conventional","any",0.284,0.615),
    ("first_calving_age","gelbvieh","Lf L",2024,"Conventional","any",23.000,35.000),
    ("milk_yield","gelbvieh","Lf L",2024,"Conventional","any",2070.000,9100.000),
    ("fat_content","gelbvieh","Lf L",2024,"Conventional","any",0.037,0.047),
    ("protein_content","gelbvieh","Lf L",2024,"Conventional","any",0.030,0.039),
    ("attrition_rate","vorwalder","Lf L",2024,"Conventional","large",0.100,0.310),
    ("attrition_rate","vorwalder","Lf L",2024,"Conventional","medium",0,0.310),
    ("attrition_rate","vorwalder","Lf L",2024,"Conventional","small",0,0.310),
    ("loss_rate","vorwalder","Lf L",2024,"Conventional","large",0.001,0.060),
    ("loss_rate","vorwalder","Lf L",2024,"Conventional","medium",0,0.060),
    ("loss_rate","vorwalder","Lf L",2024,"Conventional","small",0,0.060),
    ("intercalving_period","vorwalder","Lf L",2024,"Conventional","any",340,430),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Conventional","large",0.001,0.060),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Conventional","medium",0,0.060),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Conventional","small",0,0.060),
    ("weaning_age","vorwalder","Lf L",2024,"Conventional","any",4,12),
    ("live_weight","vorwalder","Lf L",2024,"Conventional","any",525,675),
    ("slaughter_weight","vorwalder","Lf L",2024,"Conventional","any",237,330),
    ("gutting_rate","vorwalder","Lf L",2024,"Conventional","any",0.287,0.622),
    ("first_calving_age","vorwalder","Lf L",2024,"Conventional","any",26,41),
    ("milk_yield","vorwalder","Lf L",2024,"Conventional","any",2500,8500),
    ("fat_content","vorwalder","Lf L",2024,"Conventional","any",0.034,0.043),
    ("protein_content","vorwalder","Lf L",2024,"Conventional","any",0.030,0.039),
    ("attrition_rate","jersey","Lf L",2024,"Conventional","large",0.150,0.400),
    ("attrition_rate","jersey","Lf L",2024,"Conventional","medium",0.000,0.400),
    ("attrition_rate","jersey","Lf L",2024,"Conventional","small",0.000,0.400),
    ("loss_rate","jersey","Lf L",2024,"Conventional","large",0.029,0.079),
    ("loss_rate","jersey","Lf L",2024,"Conventional","medium",0,0.079),
    ("loss_rate","jersey","Lf L",2024,"Conventional","small",0,0.079),
    ("intercalving_period","jersey","Lf L",2024,"Conventional","any",370,444),
    ("calf_loss_rate","jersey","Lf L",2024,"Conventional","large",0.041,0.180),
    ("calf_loss_rate","jersey","Lf L",2024,"Conventional","medium",0,0.180),
    ("calf_loss_rate","jersey","Lf L",2024,"Conventional","small",0,0.180),
    ("weaning_age","jersey","Lf L",2024,"Conventional","any",4,12),
    ("live_weight","jersey","Lf L",2024,"Conventional","any",370,530),
    ("slaughter_weight","jersey","Lf L",2024,"Conventional","any",80,320),
    ("gutting_rate","jersey","Lf L",2024,"Conventional","any",0.250,0.570),
    ("first_calving_age","jersey","Lf L",2024,"Conventional","any",24,38),
    ("milk_yield","jersey","Lf L",2024,"Conventional","any",3000,10000),
    ("fat_content","jersey","Lf L",2024,"Conventional","any",0.049,0.060),
    ("protein_content","jersey","Lf L",2024,"Conventional","any",0.035,0.046),
    ("fat_content","schwarzbunt","Lf L",2024,"Organic","any",0.0360,0.0490),
    ("fat_content","fleckvieh","Lf L",2024,"Organic","any",0.0360,0.0470),
    ("first_calving_age","fleckvieh","Lf L",2024,"Organic","any",23,35),
    ("first_calving_age","schwarzbunt","Lf L",2024,"Organic","any",21,33),
    ("live_weight","fleckvieh","Lf L",2024,"Organic","any",650,800),
    ("live_weight","schwarzbunt","Lf L",2024,"Organic","any",594,722),
    ("protein_content","fleckvieh","Lf L",2024,"Organic","any",0.0310,0.0360),
    ("protein_content","schwarzbunt","Lf L",2024,"Organic","any",0.0300,0.0360),
    ("slaughter_weight","fleckvieh","Lf L",2024,"Organic","any",299,404),
    ("slaughter_weight","schwarzbunt","Lf L",2024,"Organic","any",250,365),
    ("weaning_age","schwarzbunt","Lf L",2024,"Organic","any",6,18),
    ("fat_content","braunvieh","Lf L",2024,"Organic","any",0.0360,0.0470),
    ("weaning_age","fleckvieh","Lf L",2024,"Organic","any",6,16),
    ("first_calving_age","braunvieh","Lf L",2024,"Organic","any",25,37),
    ("protein_content","braunvieh","Lf L",2024,"Organic","any",0.0320,0.0370),
    ("slaughter_weight","braunvieh","Lf L",2024,"Organic","any",249,362),
    ("weaning_age","braunvieh","Lf L",2024,"Organic","any",6,16),
    ("attrition_rate","braunvieh","Lf L",2024,"Organic","large",0.0930,0.4000),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Organic","large",0.0400,0.3800),
    ("attrition_rate","fleckvieh","Lf L",2024,"Organic","large",0.0800,0.4000),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Organic","large",0.0090,0.1550),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Organic","large",0.0080,0.1500),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Organic","large",0.0010,0.1450),
    ("gutting_rate","fleckvieh","Lf L",2024,"Organic","large",0.2870,0.6220),
    ("gutting_rate","schwarzbunt","Lf L",2024,"Organic","large",0.2700,0.5900),
    ("gutting_rate","braunvieh","Lf L",2024,"Organic","large",0.2660,0.5760),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Organic","large",370,470),
    ("intercalving_period","braunvieh","Lf L",2024,"Organic","large",365,465),
    ("intercalving_period","fleckvieh","Lf L",2024,"Organic","large",355,460),
    ("live_weight","braunvieh","Lf L",2024,"Organic","large",550,750),
    ("loss_rate","schwarzbunt","Lf L",2024,"Organic","large",0.0100,0.0810),
    ("loss_rate","braunvieh","Lf L",2024,"Organic","large",0.0000,0.0500),
    ("loss_rate","fleckvieh","Lf L",2024,"Organic","large",0.0000,0.0500),
    ("milk_yield","schwarzbunt","Lf L",2024,"Organic","large",4000,11800),
    ("milk_yield","fleckvieh","Lf L",2024,"Organic","large",3665,10700),
    ("milk_yield","braunvieh","Lf L",2024,"Organic","large",4300,9500),
    ("attrition_rate","braunvieh","Lf L",2024,"Organic","medium",0.0000,0.4400),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Organic","medium",0.0000,0.1480),
    ("attrition_rate","fleckvieh","Lf L",2024,"Organic","medium",0.0000,0.4000),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Organic","medium",0.0000,0.3750),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Organic","medium",0.0000,0.1550),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Organic","medium",0.0000,0.1380),
    ("gutting_rate","fleckvieh","Lf L",2024,"Organic","medium",0.2870,0.6220),
    ("gutting_rate","schwarzbunt","Lf L",2024,"Organic","medium",0.2700,0.5900),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Organic","medium",365,485),
    ("gutting_rate","braunvieh","Lf L",2024,"Organic","medium",0.2660,0.5760),
    ("intercalving_period","fleckvieh","Lf L",2024,"Organic","medium",355,465),
    ("loss_rate","schwarzbunt","Lf L",2024,"Organic","medium",0.0000,0.0680),
    ("intercalving_period","braunvieh","Lf L",2024,"Organic","medium",365,480),
    ("loss_rate","fleckvieh","Lf L",2024,"Organic","medium",0.0000,0.0480),
    ("milk_yield","schwarzbunt","Lf L",2024,"Organic","medium",3450,10400),
    ("milk_yield","fleckvieh","Lf L",2024,"Organic","medium",3450,9765),
    ("live_weight","braunvieh","Lf L",2024,"Organic","medium",550,750),
    ("loss_rate","braunvieh","Lf L",2024,"Organic","medium",0.0000,0.0500),
    ("milk_yield","braunvieh","Lf L",2024,"Organic","medium",4200,9400),
    ("attrition_rate","braunvieh","Lf L",2024,"Organic","small",0.0000,0.4400),
    ("attrition_rate","fleckvieh","Lf L",2024,"Organic","small",0.0000,0.4000),
    ("attrition_rate","schwarzbunt","Lf L",2024,"Organic","small",0.0000,0.3750),
    ("calf_loss_rate","schwarzbunt","Lf L",2024,"Organic","small",0.0000,0.1660),
    ("calf_loss_rate","braunvieh","Lf L",2024,"Organic","small",0.0000,0.1650),
    ("calf_loss_rate","fleckvieh","Lf L",2024,"Organic","small",0.0000,0.1530),
    ("gutting_rate","fleckvieh","Lf L",2024,"Organic","small",0.4000,0.6220),
    ("gutting_rate","schwarzbunt","Lf L",2024,"Organic","small",0.3750,0.5900),
    ("gutting_rate","braunvieh","Lf L",2024,"Organic","small",0.3720,0.5760),
    ("intercalving_period","braunvieh","Lf L",2024,"Organic","small",360,500),
    ("intercalving_period","fleckvieh","Lf L",2024,"Organic","small",355,490),
    ("intercalving_period","schwarzbunt","Lf L",2024,"Organic","small",360,485),
    ("live_weight","braunvieh","Lf L",2024,"Organic","small",550,750),
    ("loss_rate","braunvieh","Lf L",2024,"Organic","small",0.0000,0.0740),
    ("loss_rate","schwarzbunt","Lf L",2024,"Organic","small",0.0000,0.0720),
    ("loss_rate","fleckvieh","Lf L",2024,"Organic","small",0.0000,0.0480),
    ("milk_yield","schwarzbunt","Lf L",2024,"Organic","small",3000,9900),
    ("milk_yield","fleckvieh","Lf L",2024,"Organic","small",2750,9350),
    ("milk_yield","braunvieh","Lf L",2024,"Organic","small",3800,9022),
    ("attrition_rate","gelbvieh","Lf L",2024,"Organic","large",0.093,0.262),
    ("attrition_rate","gelbvieh","Lf L",2024,"Organic","medium",0.000,0.262),
    ("attrition_rate","gelbvieh","Lf L",2024,"Organic","small",0.000,0.262),
    ("loss_rate","gelbvieh","Lf L",2024,"Organic","large",0.003,0.039),
    ("loss_rate","gelbvieh","Lf L",2024,"Organic","medium",0.000,0.039),
    ("loss_rate","gelbvieh","Lf L",2024,"Organic","small",0.000,0.039),
    ("intercalving_period","gelbvieh","Lf L",2024,"Organic","any",370.000,430.000),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Organic","small",0.000,0.130),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Organic","medium",0.000,0.130),
    ("calf_loss_rate","gelbvieh","Lf L",2024,"Organic","large",0.020,0.130),
    ("weaning_age","gelbvieh","Lf L",2024,"Organic","any",6.000,18.000),
    ("live_weight","gelbvieh","Lf L",2024,"Organic","any",650.000,800.000),
    ("slaughter_weight","gelbvieh","Lf L",2024,"Organic","any",280.000,369.000),
    ("gutting_rate","gelbvieh","Lf L",2024,"Organic","any",0.284,0.615),
    ("first_calving_age","gelbvieh","Lf L",2024,"Organic","any",23.000,35.000),
    ("milk_yield","gelbvieh","Lf L",2024,"Organic","any",2070.000,9100.000),
    ("fat_content","gelbvieh","Lf L",2024,"Organic","any",0.035,0.045),
    ("protein_content","gelbvieh","Lf L",2024,"Organic","any",0.030,0.040),
    ("attrition_rate","vorwalder","Lf L",2024,"Organic","large",0.100,0.310),
    ("attrition_rate","vorwalder","Lf L",2024,"Organic","medium",0,0.310),
    ("attrition_rate","vorwalder","Lf L",2024,"Organic","small",0,0.310),
    ("loss_rate","vorwalder","Lf L",2024,"Organic","large",0.001,0.060),
    ("loss_rate","vorwalder","Lf L",2024,"Organic","medium",0,0.060),
    ("loss_rate","vorwalder","Lf L",2024,"Organic","small",0,0.060),
    ("intercalving_period","vorwalder","Lf L",2024,"Organic","any",340,430),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Organic","large",0.001,0.060),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Organic","medium",0,0.060),
    ("calf_loss_rate","vorwalder","Lf L",2024,"Organic","small",0,0.060),
    ("weaning_age","vorwalder","Lf L",2024,"Organic","any",4,18),
    ("live_weight","vorwalder","Lf L",2024,"Organic","any",525,675),
    ("slaughter_weight","vorwalder","Lf L",2024,"Organic","any",237,330),
    ("gutting_rate","vorwalder","Lf L",2024,"Organic","any",0.287,0.622),
    ("first_calving_age","vorwalder","Lf L",2024,"Organic","any",26,41),
    ("milk_yield","vorwalder","Lf L",2024,"Organic","any",2500,8500),
    ("fat_content","vorwalder","Lf L",2024,"Organic","any",0.034,0.043),
    ("protein_content","vorwalder","Lf L",2024,"Organic","any",0.030,0.039),
    ("attrition_rate","jersey","Lf L",2024,"Organic","large",0.150,0.400),
    ("attrition_rate","jersey","Lf L",2024,"Organic","medium",0.000,0.400),
    ("attrition_rate","jersey","Lf L",2024,"Organic","small",0.000,0.400),
    ("loss_rate","jersey","Lf L",2024,"Organic","large",0.029,0.079),
    ("loss_rate","jersey","Lf L",2024,"Organic","medium",0,0.079),
    ("loss_rate","jersey","Lf L",2024,"Organic","small",0,0.079),
    ("intercalving_period","jersey","Lf L",2024,"Organic","any",370.000,444.000),
    ("calf_loss_rate","jersey","Lf L",2024,"Organic","large",0.041,0.180),
    ("calf_loss_rate","jersey","Lf L",2024,"Organic","medium",0,0.180),
    ("calf_loss_rate","jersey","Lf L",2024,"Organic","small",0,0.180),
    ("weaning_age","jersey","Lf L",2024,"Organic","any",6,18),
    ("live_weight","jersey","Lf L",2024,"Organic","any",370,530),
    ("slaughter_weight","jersey","Lf L",2024,"Organic","any",80.000,320.000),
    ("gutting_rate","jersey","Lf L",2024,"Organic","any",0.250,0.570),
    ("first_calving_age","jersey","Lf L",2024,"Organic","any",24,38),
    ("milk_yield","jersey","Lf L",2024,"Organic","any",3000,10000),
    ("fat_content","jersey","Lf L",2024,"Organic","any",0.049,0.060),
    ("protein_content","jersey","Lf L",2024,"Organic","any",0.035,0.046),
]


SOFTBOUNDS = pd.DataFrame(_softbounds_rows, columns=[
    "datapoint","breed","product","year","farming_method","herd_group","soft_min","soft_max"
])

# YOY rules
YOY_RULES = {
    "breed":              (False, "binary", None),
    "farming_method":     (False, "binary", None),
    "milk_yield":         (True,  "pct", 0.20),
    "milk_yield_adjusted":(True,  "pct", 0.20),
    "protein_content":    (True,  "pct", 0.08),
    "fat_content":        (True,  "pct", 0.08),
    "herd_size":          (True,  "herd_formula", None),
    "attrition_rate":     (True,  "abs", 0.08),
    "loss_rate":          (True,  "abs", 0.02),
    "live_weight":        (False, "pct", 0.10),
    "slaughter_weight":   (False, "pct", 0.10),
    "gutting_rate":       (False, "pct", 0.10),
    "grazing_days":       (False, "abs", 30),
    "intercalving_period":(False, "pct", 0.10),
    "first_calving_age":  (False, "pct", 0.10),
    "weaning_age":        (False, "abs", 2),
    "calf_loss_rate":     (True,  "abs", 0.02),
    "manure_type":        (False, "binary", None),
    "energy_consumption_per_cow": (True, "pct", 0.15),
    "manure_storage_method": (True, "manure_storage_formula", None),
    "energy_sources": (True, "energy_sources_formula", None),
}

CAUTION_POINTS = {
    "breed":2,
    "farming_method":1,
    "milk_yield":8,
    "milk_yield_adjusted":8,
    "protein_content":4,
    "fat_content":4,
    "herd_size":1,
    "attrition_rate":4,
    "loss_rate":2,
    "live_weight":2,
    "slaughter_weight":1,
    "gutting_rate":1,
    "grazing_days":2,
    "intercalving_period":1,
    "first_calving_age":2,
    "weaning_age":1,
    "calf_loss_rate":1,
    "manure_type":4,
    "energy_consumption_per_cow":2,
    "manure_storage_method":4,
    "energy_sources":1,
}

RELEVANT_COLUMNS = ["milk_yield_adjusted","assessment_id","farm_id","corporation","year","breed","farming_method",
    "milk_yield","protein_content","fat_content","herd_size","attrition_rate","loss_rate",
    "live_weight","slaughter_weight","gutting_rate","grazing_days","intercalving_period",
    "first_calving_age","weaning_age","calf_loss_rate","manure_type","energy_consumption_per_cow",
    "manure_storage_method","energy_sources"
]

NUMERIC_COLS = {
    "milk_yield_adjusted",
    "milk_yield","protein_content","fat_content","herd_size","attrition_rate","loss_rate",
    "live_weight","slaughter_weight","gutting_rate","grazing_days","intercalving_period",
    "first_calving_age","weaning_age","calf_loss_rate","energy_consumption_per_cow"
}
# Note: manure_storage_method and energy_sources are synthetic trigger columns, not regular numeric columns

# --- Normalization helpers ---
def _norm_method(val):
    if pd.isna(val):
        return None
    t = str(val).strip().lower()
    organic = {"organic","bio","ökologisch","oekologisch","biologisch"}
    conventional = {"conventional","conv","konventionell","konv","non-organic","nicht bio","standard","std"}
    if t in organic:
        return "Organic"
    if t in conventional:
        return "Conventional"
    # partial matches
    if "bio" in t or "öko" in t:
        return "Organic"
    if "konv" in t or "conv" in t or "nicht bio" in t:
        return "Conventional"
    return None

# --- YoY helpers ---
def herd_change_is_significant(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0:
        return False
    threshold = 0.06 + (0.50 * math.exp(-prev/50.0))
    return abs(curr - prev) / abs(prev) > threshold


def manure_storage_change_is_significant(curr_row, prev_row):
    """Check if sum of absolute changes across all manure storage types is >30%"""
    storage_columns = [
        "manure_storage_open",
        "manure_storage_straw_floating_cover", 
        "manure_storage_foil_floating_cover",
        "manure_storage_natural_floating_cover",
        "manure_storage_fixed_cover",
        "manure_storage_slatted_floor",
        "manure_storage_biogas_airtight",
        "manure_storage_biogas_not_airtight"
    ]
    
    # Check if any of the required columns exist
    has_storage_data = False
    for col in storage_columns:
        if (curr_row is not None and col in curr_row and pd.notna(curr_row.get(col))) or \
           (prev_row is not None and col in prev_row and pd.notna(prev_row.get(col))):
            has_storage_data = True
            break
    
    if not has_storage_data:
        return False  # No storage data available
    
    total_abs_change = 0.0
    
    for col in storage_columns:
        curr_val = curr_row.get(col, 0) if curr_row is not None else 0
        prev_val = prev_row.get(col, 0) if prev_row is not None else 0
        
        # Convert to numeric, defaulting to 0 if not numeric
        try:
            curr_val = float(curr_val) if pd.notna(curr_val) else 0.0
            prev_val = float(prev_val) if pd.notna(prev_val) else 0.0
        except (ValueError, TypeError):
            curr_val = 0.0
            prev_val = 0.0
            
        total_abs_change += abs(curr_val - prev_val)
    
    # Check if sum of absolute changes > 30% (0.30)
    return total_abs_change > 0.30


def energy_sources_change_is_significant(curr_row, prev_row):
    """Check if sum of absolute changes across all energy source types is >40%"""
    energy_source_columns = [
        "energy_source_grid",
        "energy_source_biogas_crops",
        "energy_source_biogas_manure", 
        "energy_source_green_electricity",
        "energy_source_photovoltaic",
        "energy_source_wind_on_land"
    ]
    
    # Check if any of the required columns exist
    has_energy_data = False
    for col in energy_source_columns:
        if (curr_row is not None and col in curr_row and pd.notna(curr_row.get(col))) or \
           (prev_row is not None and col in prev_row and pd.notna(prev_row.get(col))):
            has_energy_data = True
            break
    
    if not has_energy_data:
        return False  # No energy source data available
    
    total_abs_change = 0.0
    
    for col in energy_source_columns:
        curr_val = curr_row.get(col, 0) if curr_row is not None else 0
        prev_val = prev_row.get(col, 0) if prev_row is not None else 0
        
        # Convert to numeric, defaulting to 0 if not numeric
        try:
            curr_val = float(curr_val) if pd.notna(curr_val) else 0.0
            prev_val = float(prev_val) if pd.notna(prev_val) else 0.0
        except (ValueError, TypeError):
            curr_val = 0.0
            prev_val = 0.0
            
        total_abs_change += abs(curr_val - prev_val)
    
    # Check if sum of absolute changes > 40% (0.40)
    return total_abs_change > 0.40


def change_is_significant(col, curr, prev, herd_prev=None, curr_row=None, prev_row=None):
    _, ctype, tv = YOY_RULES[col]
    
    # Special handling for formula-based checks that don't depend on curr/prev values
    if ctype == "manure_storage_formula":
        return manure_storage_change_is_significant(curr_row, prev_row)
    if ctype == "energy_sources_formula":
        return energy_sources_change_is_significant(curr_row, prev_row)
    
    # Regular checks that require valid curr/prev values
    if pd.isna(prev) or pd.isna(curr):
        return False
    
    if ctype == "binary":
        return curr != prev
    if ctype == "pct":
        if prev == 0:
            return curr != 0
        return abs(curr - prev) / abs(prev) > tv
    if ctype == "abs":
        return abs(curr - prev) > tv
    if ctype == "herd_formula":
        return herd_change_is_significant(curr, prev)
    return False


def flag_if_same(col):
    return YOY_RULES[col][0]


def format_change(curr, prev):
    if pd.isna(prev):
        return ""
    try:
        if isinstance(curr, (int,float,np.floating)) and isinstance(prev, (int,float,np.floating)) and prev != 0:
            pct = (curr - prev)/prev*100
            return f"Significant change from {prev} to {curr} ({pct:+.1f}%)"
        return f"Significant change from {prev} to {curr}"
    except Exception:
        return f"Significant change from {prev} to {curr}"


def categorize_points(points):
    if points >= 15:
        return "Significant Flags (In Review)"
    elif points >= 8:
        return "Major Flags (In Review)"
    elif points > 0:
        return "Minor Flags (Complete - Auto-approved)"
    else:
        return "No Flags (Complete - Auto-approved)"

# ==========================================
# Visualization helpers
# ==========================================

def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse CDF for standard normal using Peter J. Acklam's algorithm.
    Valid for 0<p<1. """
    # Coefficients in rational approximations
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]

    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]

    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]

    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    if p <= 0 or p >= 1:
        return np.nan

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def assign_herd_group(size: float, groups_df: pd.DataFrame) -> str | None:
    if pd.isna(size):
        return None
    for _, r in groups_df.iterrows():
        gname = str(r["name"]) if pd.notna(r.get("name")) else ""
        gmin = r.get("min")
        gmax = r.get("max")
        try:
            gmin = float(gmin) if pd.notna(gmin) else -np.inf
        except Exception:
            gmin = -np.inf
        try:
            gmax = float(gmax) if pd.notna(gmax) else np.inf
        except Exception:
            gmax = np.inf
        # Inclusive bounds on both sides; avoid overlaps in definitions when editing.
        if size >= gmin and size <= gmax:
            return gname
    return None


def build_herd_groups_state():
    if "herd_groups" not in st.session_state:
        st.session_state["herd_groups"] = pd.DataFrame([
            {"name": "small",  "min": None, "max": 29},
            {"name": "medium", "min": 30,   "max": 59},
            {"name": "large",  "min": 60,   "max": None},
        ])


def _norm_colnames(df: pd.DataFrame) -> dict:
    """Return a map from lower-case names to actual column names for robust lookup."""
    return {c.lower(): c for c in df.columns}


# ==========================================
# Input
# ==========================================

uploaded = st.file_uploader("Upload CSV", type=["csv"])
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=';')

if df is None:
    st.info("Please upload a CSV to begin. The app expects the columns listed in the prompt.")
    st.stop()

# --- Basic normalization shared by both tabs ---
if "breed" not in df.columns:
    st.error("Missing required column: 'breed'")
    st.stop()

# lowercase breed and filter allowed
df["breed"] = df["breed"].astype(str).str.lower()
df = df[df["breed"].isin(ALLOWED_BREEDS)].copy()

# Normalize types
if "year" not in df.columns:
    st.error("Missing required column: 'year'")
    st.stop()

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

# Create synthetic columns for custom checks if the underlying data columns exist
manure_storage_columns = [
    "manure_storage_open", "manure_storage_straw_floating_cover", 
    "manure_storage_foil_floating_cover", "manure_storage_natural_floating_cover",
    "manure_storage_fixed_cover", "manure_storage_slatted_floor",
    "manure_storage_biogas_airtight", "manure_storage_biogas_not_airtight"
]
energy_source_columns = [
    "energy_source_grid", "energy_source_biogas_crops", "energy_source_biogas_manure",
    "energy_source_green_electricity", "energy_source_photovoltaic", "energy_source_wind_on_land"
]

# Create synthetic manure_storage_method column if any storage columns exist
manure_storage_found = [col for col in manure_storage_columns if col in df.columns]
if manure_storage_found:
    df["manure_storage_method"] = 1  # Dummy value to trigger YOY checks
else:
    df["manure_storage_method"] = np.nan

# Create synthetic energy_sources column if any energy source columns exist  
energy_source_found = [col for col in energy_source_columns if col in df.columns]
if energy_source_found:
    df["energy_sources"] = 1  # Dummy value to trigger YOY checks
else:
    df["energy_sources"] = np.nan

# Debug Analysis for Custom Checks
def analyze_storage_energy_changes(df_current, df_previous):
    """Analyze manure storage and energy source changes for debugging"""
    analyses = {"manure_storage": [], "energy_sources": []}
    
    if df_previous.empty:
        return analyses
        
    prev_lookup_debug = df_previous.sort_values("submitted_at" if "submitted_at" in df_previous.columns else "year") \
                                   .drop_duplicates(subset=["farm_id","corporation"], keep="first") \
                                   .set_index(["farm_id","corporation"])
    
    for idx, curr_row in df_current.iterrows():
        farm_id = curr_row.get("farm_id")
        corp = curr_row.get("corporation")
        assessment_id = curr_row.get("assessment_id")
        
        if (farm_id, corp) not in prev_lookup_debug.index:
            continue
        
        prev_row = prev_lookup_debug.loc[(farm_id, corp)]
        
        # Analyze manure storage changes
        if manure_storage_found:
            total_manure_change = 0.0
            for col in manure_storage_columns:
                curr_val = float(curr_row.get(col, 0)) if pd.notna(curr_row.get(col)) else 0.0
                prev_val = float(prev_row.get(col, 0)) if pd.notna(prev_row.get(col)) else 0.0
                total_manure_change += abs(curr_val - prev_val)
            
            analyses["manure_storage"].append({
                "assessment_id": assessment_id,
                "farm_id": farm_id,
                "corporation": corp,
                "total_change": total_manure_change,
                "exceeds_threshold": total_manure_change > 0.30,
                "details": {col: {"curr": float(curr_row.get(col, 0)) if pd.notna(curr_row.get(col)) else 0.0, 
                                 "prev": float(prev_row.get(col, 0)) if pd.notna(prev_row.get(col)) else 0.0} 
                           for col in manure_storage_columns}
            })
        
        # Analyze energy source changes
        if energy_source_found:
            total_energy_change = 0.0
            for col in energy_source_columns:
                curr_val = float(curr_row.get(col, 0)) if pd.notna(curr_row.get(col)) else 0.0
                prev_val = float(prev_row.get(col, 0)) if pd.notna(prev_row.get(col)) else 0.0
                total_energy_change += abs(curr_val - prev_val)
            
            analyses["energy_sources"].append({
                "assessment_id": assessment_id,
                "farm_id": farm_id,
                "corporation": corp,
                "total_change": total_energy_change,
                "exceeds_threshold": total_energy_change > 0.40,
                "details": {col: {"curr": float(curr_row.get(col, 0)) if pd.notna(curr_row.get(col)) else 0.0, 
                                 "prev": float(prev_row.get(col, 0)) if pd.notna(prev_row.get(col)) else 0.0} 
                           for col in energy_source_columns}
            })
    
    return analyses

# Display debug info  
with st.expander("🔍 Custom Checks Debug Analysis", expanded=False):
    st.write("### Found Columns:")
    if manure_storage_found:
        st.write(f"**Manure Storage:** {manure_storage_found}")
        st.write("✅ Manure storage YOY check is ENABLED")
    else:
        st.write("❌ No manure storage columns found")
        
    if energy_source_found:
        st.write(f"**Energy Sources:** {energy_source_found}")
        st.write("✅ Energy sources YOY check is ENABLED")
    else:
        st.write("❌ No energy source columns found")
    
    st.write("### Thresholds:")
    st.write("- Manure Storage: >30% (0.30)")
    st.write("- Energy Sources: >40% (0.40)")

# Normalize farming method and coerce numerics used in checks
if "farming_method" in df.columns:
    df["farming_method_norm"] = df["farming_method"].apply(_norm_method)
else:
    df["farming_method_norm"] = None

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute milk_yield_adjusted based on milk_yield_type and calf_rearing_method
def _compute_milk_yield_adjusted(row):
    try:
        base = row.get("milk_yield")
        if pd.isna(base):
            return np.nan
        t = str(row.get("milk_yield_type")).strip().lower() if row.get("milk_yield_type") is not None else ""
        if t == "producedvolume":
            return base
        if t == "soldvolume":
            crm = str(row.get("calf_rearing_method")).strip().lower() if row.get("calf_rearing_method") is not None else ""
            if crm == "whole_milk":
                return base + 420
            if crm == "mat":
                return base + 50
            if crm == "milk_mat_mix":
                return base + 230
            # Fallback for unknown methods under soldVolume: no adjustment
            return base
        # Unknown type: fallback to base value
        return base
    except Exception:
        return np.nan

if "milk_yield" in df.columns:
    try:
        df["milk_yield_adjusted"] = df.apply(_compute_milk_yield_adjusted, axis=1)
    except Exception:
        df["milk_yield_adjusted"] = np.nan

if df.empty:
    st.warning("No rows left after filtering by allowed breeds.")
    st.stop()

# ==========================================
# Visualization
# ==========================================

st.subheader("Distribution Visualizer")

# --- Herd size groups editor ---
build_herd_groups_state()
with st.expander("Herd size groups (edit & add)", expanded=False):
    st.caption("Default: small (<=29), medium (30–59), large (>=60). You can add/edit groups; bounds are inclusive.")
    edited = st.data_editor(
        st.session_state["herd_groups"],
        num_rows="dynamic",
        width='stretch',
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Group Name", required=True),
            "min": st.column_config.NumberColumn("Min"),
            "max": st.column_config.NumberColumn("Max"),
        },
        key="herd_groups_editor",
    )
    # Persist edits
    st.session_state["herd_groups"] = edited

# --- Controls ---
col_f1, col_f2, col_f3, col_f4 = st.columns([1,1,1,1])
with col_f1:
    # Datapoint selector
    dp_map = {
        "Milk yield (adjusted)": "milk_yield_adjusted",
        "Slaughter weight": "slaughter_weight",
        "Milk yield": "milk_yield",
        "Live weight": "live_weight",
        "Feed total": "feed_total",
        "Calf loss rate": "calf_loss_rate",
        "Loss rate": "loss_rate",
        "First calving age": "first_calving_age",
        "Attrition rate": "attrition_rate",
        "Protein content": "protein_content",
        "Fat content": "fat_content",
        "Gutting rate": "gutting_rate",
        "Intercalving period": "intercalving_period",
        "Weaning age": "weaning_age",
        "Energy per cow": "energy_consumption_per_cow",
    }
    dp_label = st.selectbox("Datapoint", list(dp_map.keys()), index=1)
    dp_col = dp_map[dp_label]
with col_f2:
    # Corporation selector
    corp_options = sorted(df["corporation"].dropna().unique().tolist()) if "corporation" in df.columns else []
    corp_label_options = ["All corporations"] + corp_options
    corp_sel = st.selectbox("Corporation", corp_label_options, index=0)
with col_f3:
    # Year selector for visualization
    year_sel_mode = st.selectbox("Year", ["2023 and 2024", "Only 2024"], index=0)
with col_f4:
    st.markdown("Bounds settings")
    show_mean = st.checkbox("Show mean", value=True)
    show_median = st.checkbox("Show median", value=False)
    show_quantiles = st.checkbox("Show quantile bounds", value=False)
    q_low = st.number_input("Lower quantile (%)", min_value=0.0, max_value=100.0, value=2.5, step=0.1, key="q_low_all")
    q_high = st.number_input("Upper quantile (%)", min_value=0.0, max_value=100.0, value=97.5, step=0.1, key="q_high_all")
    show_std = st.checkbox("Show std deviation bounds", value=False)
    std_k = st.number_input("# of SD", min_value=0.0, max_value=10.0, value=0.5, step=0.1, key="std_k_all")
    show_iqr = st.checkbox("Show IQR outlier bounds (Tukey's rule)", value=False)
    iqr_multiplier = st.number_input("IQR fence multiplier", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="iqr_multiplier_all")
    show_standard_values = st.checkbox("Show standard values", value=False)
    show_softbounds = st.checkbox("Show softbounds", value=True)
    show_strict_bounds = st.checkbox("Show adjusted bounds (range-based)", value=True)
    strict_pct = st.number_input("Adjusted bounds % (+ narrows, - broadens)", min_value=-50.0, max_value=50.0, value=10.0, step=1.0, key="strict_pct_all", 
                                  help="Adjusts bounds: Positive % narrows range (e.g., +10% makes bounds 10% stricter). Negative % broadens range (e.g., -10% makes bounds 10% wider).")
    show_multiplicative_bounds = st.checkbox("Show multiplicative bounds (value-based)", value=False)
    mult_pct = st.number_input("Multiplicative bounds %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="mult_pct_all",
                                help="Multiplies each bound by percentage: upper bound × (1+%) and lower bound × (1-%). E.g., 10% makes lower bound 10% smaller and upper bound 10% larger.")
    # Global histogram bin control and x-range controls
    bins_all = st.slider("Histogram bins (all plots)", min_value=5, max_value=150, value=90, step=1, key="bins_all")
    # Compute defaults for x-range based on corporation/year filtered data
    _defaults_df = df.copy()
    if corp_sel != "All corporations" and "corporation" in _defaults_df.columns:
        _defaults_df = _defaults_df[_defaults_df["corporation"] == corp_sel]
    if "year" in _defaults_df.columns:
        if year_sel_mode == "Only 2024":
            _defaults_df = _defaults_df[_defaults_df["year"] == 2024]
        else:
            _defaults_df = _defaults_df[_defaults_df["year"].isin([2023, 2024])]
    _x_defaults = pd.to_numeric(_defaults_df.get(dp_col, pd.Series(dtype=float)), errors="coerce").dropna()
    try:
        _x_min_default = float(_x_defaults.min()) if not _x_defaults.empty else 0.0
        _x_max_default = float(_x_defaults.max()) if not _x_defaults.empty else 1.0
    except Exception:
        _x_min_default, _x_max_default = 0.0, 1.0
    use_xrange = st.checkbox("Set x-axis range (all plots)", value=False, key="use_xrange_all")
    if use_xrange:
        x_min_all = st.number_input("X min", value=_x_min_default, key="x_min_all")
        x_max_all = st.number_input("X max", value=_x_max_default, key="x_max_all")
    else:
        x_min_all, x_max_all = None, None

# Compute helper columns available for visualization only
colmap = _norm_colnames(df)
# Feed total
rough_col = colmap.get("roughages")
conc_col  = colmap.get("concentrates")
juice_col = colmap.get("juices")
if dp_col == "feed_total":
    if rough_col and conc_col and juice_col:
        df["feed_total"] = df[rough_col].astype(float) + df[conc_col].astype(float) + df[juice_col].astype(float)
    else:
        df["feed_total"] = np.nan
        st.warning("'Feed total' needs columns 'roughages', 'concentrates', and 'juices'. Not all were found.")
# Energy per cow: expect column named 'energy_consumption_per_cow'
if dp_col == "energy_consumption_per_cow" and "energy_consumption_per_cow" not in df.columns:
    st.info("Column 'energy_consumption_per_cow' not found – this metric will be empty unless present in your data.")
    df["energy_consumption_per_cow"] = np.nan

# Assign herd groups
hg_df = st.session_state["herd_groups"].copy()
if "herd_size" in df.columns:
    df["herd_group"] = df["herd_size"].apply(lambda s: assign_herd_group(s, hg_df))
else:
    df["herd_group"] = None

# Base filters: corporation and year
vis_df = df.copy()
# Apply corporation filter if a specific one is selected
if corp_sel != "All corporations" and "corporation" in vis_df.columns:
    vis_df = vis_df[vis_df["corporation"] == corp_sel]
# Apply year filter as requested
if "year" in vis_df.columns:
    if year_sel_mode == "Only 2024":
        vis_df = vis_df[vis_df["year"] == 2024]
    else:
        vis_df = vis_df[vis_df["year"].isin([2023, 2024])]
# Prepare iteration domains
fm_order = ["Conventional", "Organic"]
fms = [fm for fm in fm_order if fm in vis_df["farming_method_norm"].dropna().unique().tolist()]
breeds = [b for b in ["braunvieh", "fleckvieh", "schwarzbunt"] if b in vis_df["breed"].dropna().unique().tolist()]
# Herd groups list + All
herd_groups = st.session_state["herd_groups"]["name"].dropna().astype(str).tolist() if "herd_groups" in st.session_state else []
herd_groups = herd_groups[:3] if len(herd_groups) > 3 else herd_groups
herd_groups_all = herd_groups + ["All"]

# Datasource mapping for standard values
DATASOURCE_CANDIDATES = {
    "milk_yield": ["datasource_milk_yield"],
    "calf_loss_rate": ["datasource_kaelberverlustquote", "datasource_kälberverlustquote"],
    "live_weight": ["datasource_live_weight", "datasource_lebendgewicht"],
    "first_calving_age": ["datasource_erstkalbealter", "datasource erstkalbealter"],
    "slaughter_weight": ["datasource_gewicht_schlachtkuh"],
    "weaning_age": ["datasource_absetzalter"],
    "intercalving_period": ["datasource_zwischenkalbezeit", "datsource_zwischenkalbezeit"],
    "loss_rate": ["datasource_kuhverluste"],
    "attrition_rate": ["datasource_abgangsquote"],
    "fat_content": ["datasource_fettgehalt_der_milch"],
    "protein_content": ["datasource_eiweissgehalt_der_milch"],
    "energy_consumption_per_cow": ["datasource_energy_consumption", "datasource_energieverbrauch"],
}

def _simple_norm_key(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

norm_any = {_simple_norm_key(c): c for c in df.columns}

for fm in fms:
    st.markdown(f"### {fm}")
    for breed in breeds:
        st.markdown(f"**Breed: {breed}**")
        for i in range(0, len(herd_groups_all), 2):
            row_hgs = herd_groups_all[i:i+2]
            cols = st.columns(2)
            for idx_col, hg_name in enumerate(row_hgs):
                with cols[idx_col]:
                    subset = vis_df[(vis_df["farming_method_norm"] == fm) & (vis_df["breed"] == breed)].copy()
                    if hg_name != "All" and "herd_group" in subset.columns:
                        subset = subset[subset["herd_group"] == hg_name]

                    x = pd.to_numeric(subset.get(dp_col, pd.Series(dtype=float)), errors="coerce").dropna()
                    N = int(x.shape[0])
                    mean = float(np.mean(x)) if N>0 else np.nan
                    sd = float(np.std(x, ddof=1)) if N>1 else np.nan
                    median = float(np.median(x)) if N>0 else np.nan
                    qL = float(np.percentile(x, q_low)) if (N>0 and show_quantiles) else np.nan
                    qU = float(np.percentile(x, q_high)) if (N>0 and show_quantiles) else np.nan
                    sdL = mean - std_k*sd if (show_std and math.isfinite(mean) and math.isfinite(sd)) else np.nan
                    sdU = mean + std_k*sd if (show_std and math.isfinite(mean) and math.isfinite(sd)) else np.nan
                    # IQR bounds (Tukey's rule: Q1 - k*IQR and Q3 + k*IQR, where k is configurable)
                    q1 = float(np.percentile(x, 25)) if (N>0 and show_iqr) else np.nan
                    q3 = float(np.percentile(x, 75)) if (N>0 and show_iqr) else np.nan
                    iqr = q3 - q1 if (show_iqr and math.isfinite(q1) and math.isfinite(q3)) else np.nan
                    iqrL = q1 - iqr_multiplier * iqr if (show_iqr and math.isfinite(iqr)) else np.nan
                    iqrU = q3 + iqr_multiplier * iqr if (show_iqr and math.isfinite(iqr)) else np.nan

                    # Build histogram with adjustable ranges and normal curve overlay
                    fig = go.Figure()
                    if N > 0:
                        # Compute counts for y-axis scaling and curve overlay
                        try:
                            counts, edges = np.histogram(x, bins=bins_all, density=False)
                            y_max_hist = float(np.max(counts)) if counts.size > 0 else 0.0
                            bin_width_avg = float(np.mean(np.diff(edges))) if edges.size > 1 else (float(np.max(x) - np.min(x)) / float(bins_all) if (N>0 and bins_all>0 and (np.max(x) > np.min(x))) else 1.0)
                        except Exception:
                            counts, edges = np.array([]), np.array([])
                            y_max_hist = 0.0
                            bin_width_avg = 1.0

                        # Histogram trace (counts) with adjustable bins
                        fig.add_histogram(x=x, nbinsx=bins_all, opacity=0.6, name='Distribution', hovertemplate='x=%{x}<br>y=%{y}<extra></extra>')

                        # Normal fit scaled to counts
                        y_max_pdf = 0.0
                        if math.isfinite(mean) and math.isfinite(sd) and sd > 0:
                            # Use custom x-range if provided; otherwise subset min/max
                            _xp_min = float(np.min(x))
                            _xp_max = float(np.max(x))
                            if use_xrange and x_min_all is not None and x_max_all is not None and x_max_all > x_min_all:
                                _xp_min, _xp_max = float(x_min_all), float(x_max_all)
                            xp = np.linspace(_xp_min, _xp_max, 200)
                            pdf = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((xp-mean)/sd)**2)
                            pdf = pdf * N * bin_width_avg
                            y_max_pdf = float(np.max(pdf)) if pdf.size > 0 else 0.0
                            fig.add_trace(go.Scatter(x=xp, y=pdf, mode='lines', name='Normal fit', hoverinfo='skip'))

                        y_max_line = max(y_max_hist, y_max_pdf, 1.0)
                        vline_y = [0, y_max_line]
                        # Apply global x-axis range if enabled
                        if use_xrange and x_min_all is not None and x_max_all is not None and x_max_all > x_min_all:
                            fig.update_xaxes(range=[x_min_all, x_max_all])
                        if show_mean and math.isfinite(mean):
                            fig.add_trace(go.Scatter(x=[mean, mean], y=vline_y, mode='lines', name=f"Mean = {mean:.3f}", line=dict(color='black', width=2), hoverinfo='skip'))
                        if show_median and math.isfinite(median):
                            fig.add_trace(go.Scatter(x=[median, median], y=vline_y, mode='lines', name=f"Median = {median:.3f}", line=dict(color='dimgray', width=2, dash='dash'), hoverinfo='skip'))
                        if show_quantiles:
                            if math.isfinite(qL):
                                fig.add_trace(go.Scatter(x=[qL, qL], y=vline_y, mode='lines', name=f"Quantile lower ({q_low:.1f}%) = {qL:.3f}", line=dict(width=2, dash='dot'), hoverinfo='skip'))
                            if math.isfinite(qU):
                                fig.add_trace(go.Scatter(x=[qU, qU], y=vline_y, mode='lines', name=f"Quantile upper ({q_high:.1f}%) = {qU:.3f}", line=dict(width=2, dash='dot'), hoverinfo='skip'))
                        if show_std:
                            if math.isfinite(sdL):
                                fig.add_trace(go.Scatter(x=[sdL, sdL], y=vline_y, mode='lines', name=f"SD lower (k={std_k:.2f}) = {sdL:.3f}", line=dict(color='teal', width=2, dash='dashdot'), hoverinfo='skip'))
                            if math.isfinite(sdU):
                                fig.add_trace(go.Scatter(x=[sdU, sdU], y=vline_y, mode='lines', name=f"SD upper (k={std_k:.2f}) = {sdU:.3f}", line=dict(color='teal', width=2, dash='dashdot'), hoverinfo='skip'))
                        if show_iqr:
                            if math.isfinite(iqrL):
                                fig.add_trace(go.Scatter(x=[iqrL, iqrL], y=vline_y, mode='lines', name=f"IQR lower fence (k={iqr_multiplier:.1f}) = {iqrL:.3f}", line=dict(color='red', width=2, dash='longdash'), hoverinfo='skip'))
                            if math.isfinite(iqrU):
                                fig.add_trace(go.Scatter(x=[iqrU, iqrU], y=vline_y, mode='lines', name=f"IQR upper fence (k={iqr_multiplier:.1f}) = {iqrU:.3f}", line=dict(color='red', width=2, dash='longdash'), hoverinfo='skip'))

                        # Softbounds for this fm/breed/herd_group (year 2024). Use milk_yield for adjusted
                        if show_softbounds:
                            dp_soft = "milk_yield" if dp_col == "milk_yield_adjusted" else dp_col
                            # Try herd-specific bounds first, then fall back to "any"
                            sb_specific = SOFTBOUNDS[(SOFTBOUNDS["datapoint"]==dp_soft) & (SOFTBOUNDS["breed"]==breed) & 
                                                    (SOFTBOUNDS["year"]==2024) & (SOFTBOUNDS["farming_method"]==fm) & 
                                                    (SOFTBOUNDS["herd_group"]==hg_name)]
                            sb_any = SOFTBOUNDS[(SOFTBOUNDS["datapoint"]==dp_soft) & (SOFTBOUNDS["breed"]==breed) & 
                                               (SOFTBOUNDS["year"]==2024) & (SOFTBOUNDS["farming_method"]==fm) & 
                                               (SOFTBOUNDS["herd_group"]=="any")]
                            sb_subset = sb_specific if not sb_specific.empty else sb_any
                            if not sb_subset.empty:
                                smin = float(sb_subset["soft_min"].min())
                                smax = float(sb_subset["soft_max"].max())
                                hg_label = f" ({hg_name})" if not sb_specific.empty and hg_name != "All" else ""
                                fig.add_trace(go.Scatter(x=[smin, smin], y=vline_y, mode='lines', name=f"Softbound min{hg_label} = {smin:.3f}", line=dict(color='orange', width=2, dash='dash'), hoverinfo='skip'))
                                fig.add_trace(go.Scatter(x=[smax, smax], y=vline_y, mode='lines', name=f"Softbound max{hg_label} = {smax:.3f}", line=dict(color='orange', width=2, dash='dash'), hoverinfo='skip'))
                                
                                # Adjusted bounds: narrow (positive %) or broaden (negative %) the range
                                if show_strict_bounds and strict_pct != 0:
                                    sb_range = smax - smin
                                    adjustment = sb_range * (strict_pct / 100.0)
                                    strict_min = smin + adjustment
                                    strict_max = smax - adjustment
                                    if strict_min < strict_max:  # Only show if bounds are valid
                                        direction_label = f"{'+' if strict_pct > 0 else ''}{strict_pct:.0f}%"
                                        fig.add_trace(go.Scatter(x=[strict_min, strict_min], y=vline_y, mode='lines', 
                                                                name=f"Adjusted min{hg_label} ({direction_label}) = {strict_min:.3f}", 
                                                                line=dict(color='crimson', width=2, dash='dashdot'), hoverinfo='skip'))
                                        fig.add_trace(go.Scatter(x=[strict_max, strict_max], y=vline_y, mode='lines', 
                                                                name=f"Adjusted max{hg_label} ({direction_label}) = {strict_max:.3f}", 
                                                                line=dict(color='crimson', width=2, dash='dashdot'), hoverinfo='skip'))
                                
                                # Multiplicative bounds: multiply each bound by (1±%)
                                if show_multiplicative_bounds and mult_pct > 0:
                                    mult_min = smin * (1 - mult_pct / 100.0)
                                    mult_max = smax * (1 + mult_pct / 100.0)
                                    fig.add_trace(go.Scatter(x=[mult_min, mult_min], y=vline_y, mode='lines',
                                                            name=f"Mult. min{hg_label} (×{1-mult_pct/100:.2f}) = {mult_min:.3f}",
                                                            line=dict(color='darkviolet', width=2, dash='longdashdot'), hoverinfo='skip'))
                                    fig.add_trace(go.Scatter(x=[mult_max, mult_max], y=vline_y, mode='lines',
                                                            name=f"Mult. max{hg_label} (×{1+mult_pct/100:.2f}) = {mult_max:.3f}",
                                                            line=dict(color='darkviolet', width=2, dash='longdashdot'), hoverinfo='skip'))

                        # Overlay standard value(s) for this subset (dp_col only)
                        if show_standard_values:
                            ds_cols_present = []
                            for cand in DATASOURCE_CANDIDATES.get(dp_col, []):
                                actual = colmap.get(cand.lower())
                                if actual is None:
                                    actual = norm_any.get(_simple_norm_key(cand))
                                if actual is not None:
                                    ds_cols_present.append(actual)
                            standard_vals = []
                            if ds_cols_present:
                                ds_mask = pd.Series(False, index=subset.index)
                                for c in ds_cols_present:
                                    ds_mask |= subset[c].astype(str).str.strip().str.lower() == "standard"
                                sv = pd.to_numeric(subset.loc[ds_mask, dp_col], errors='coerce').dropna().unique().tolist()
                                standard_vals = sorted([float(v) for v in sv])
                            std_colors = ['purple', 'magenta', 'brown', 'olive', 'darkgreen', 'navy', 'maroon', 'darkblue']
                            # Show ALL standard values, not just first 4
                            for si, sval in enumerate(standard_vals):
                                color_idx = si % len(std_colors)
                                suffix = f" ({si+1})" if len(standard_vals) > 1 else ""
                                fig.add_trace(go.Scatter(x=[sval, sval], y=vline_y, mode='lines', name=f"Standard{suffix} = {sval:.3f}", line=dict(color=std_colors[color_idx], width=2), hoverinfo='skip'))

                        title = f"{dp_label} | {fm} | {breed} | {'All herd sizes' if hg_name=='All' else hg_name}"
                        fig.update_layout(title=title, xaxis_title=dp_label, yaxis_title="Count", hovermode='x', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                        
                        # Add N in top right corner
                        fig.add_annotation(
                            text=f"N = {N}",
                            xref="paper", yref="paper",
                            x=0.98, y=0.95,
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                        # Per-plot summary
                        stat_rows = [
                            {"Metric": "N", "Value": N},
                            {"Metric": "Mean", "Value": None if not math.isfinite(mean) else round(mean, 3)},
                            {"Metric": "Median", "Value": None if not math.isfinite(median) else round(median, 3)},
                        ]
                        if show_quantiles:
                            stat_rows.extend([
                                {"Metric": f"Quantile lower ({q_low:.1f}%)", "Value": None if not math.isfinite(qL) else round(qL, 3)},
                                {"Metric": f"Quantile upper ({q_high:.1f}%)", "Value": None if not math.isfinite(qU) else round(qU, 3)},
                            ])
                            if math.isfinite(qL) and math.isfinite(qU):
                                below_q = int(np.sum(x < qL))
                                above_q = int(np.sum(x > qU))
                                total_q = below_q + above_q
                                pct_q = (total_q / N * 100) if N > 0 else 0.0
                                stat_rows.extend([
                                    {"Metric": "Outside quantiles (below)", "Value": below_q},
                                    {"Metric": "Outside quantiles (above)", "Value": above_q},
                                    {"Metric": "Outside quantiles (total)", "Value": total_q},
                                    {"Metric": "Outside quantiles (%)", "Value": f"{pct_q:.1f}%"},
                                ])
                        if show_std and math.isfinite(sd):
                            stat_rows.extend([
                                {"Metric": f"SD bounds (k={std_k:.2f}) lower", "Value": None if not math.isfinite(sdL) else round(sdL, 3)},
                                {"Metric": f"SD bounds (k={std_k:.2f}) upper", "Value": None if not math.isfinite(sdU) else round(sdU, 3)},
                            ])
                            if math.isfinite(sdL) and math.isfinite(sdU):
                                below_s = int(np.sum(x < sdL))
                                above_s = int(np.sum(x > sdU))
                                total_s = below_s + above_s
                                pct_s = (total_s / N * 100) if N > 0 else 0.0
                                stat_rows.extend([
                                    {"Metric": "Outside SD (below)", "Value": below_s},
                                    {"Metric": "Outside SD (above)", "Value": above_s},
                                    {"Metric": "Outside SD (total)", "Value": total_s},
                                    {"Metric": "Outside SD (%)", "Value": f"{pct_s:.1f}%"},
                                ])
                        if show_iqr and math.isfinite(iqr):
                            stat_rows.extend([
                                {"Metric": "Q1", "Value": None if not math.isfinite(q1) else round(q1, 3)},
                                {"Metric": "Q3", "Value": None if not math.isfinite(q3) else round(q3, 3)},
                                {"Metric": "IQR", "Value": None if not math.isfinite(iqr) else round(iqr, 3)},
                                {"Metric": f"IQR lower fence (k={iqr_multiplier:.1f})", "Value": None if not math.isfinite(iqrL) else round(iqrL, 3)},
                                {"Metric": f"IQR upper fence (k={iqr_multiplier:.1f})", "Value": None if not math.isfinite(iqrU) else round(iqrU, 3)},
                            ])
                            if math.isfinite(iqrL) and math.isfinite(iqrU):
                                below_iqr = int(np.sum(x < iqrL))
                                above_iqr = int(np.sum(x > iqrU))
                                total_iqr = below_iqr + above_iqr
                                pct_iqr = (total_iqr / N * 100) if N > 0 else 0.0
                                stat_rows.extend([
                                    {"Metric": "Outliers (below fence)", "Value": below_iqr},
                                    {"Metric": "Outliers (above fence)", "Value": above_iqr},
                                    {"Metric": f"Total outliers (k={iqr_multiplier:.1f})", "Value": total_iqr},
                                    {"Metric": f"Outliers (%)", "Value": f"{pct_iqr:.1f}%"},
                                ])

                        # Softbounds summary (year 2024) - herd-specific or "any"
                        if show_softbounds:
                            dp_soft = "milk_yield" if dp_col == "milk_yield_adjusted" else dp_col
                            # Try herd-specific bounds first, then fall back to "any"
                            sb_specific = SOFTBOUNDS[(SOFTBOUNDS["datapoint"]==dp_soft) & (SOFTBOUNDS["breed"]==breed) & 
                                                    (SOFTBOUNDS["year"]==2024) & (SOFTBOUNDS["farming_method"]==fm) & 
                                                    (SOFTBOUNDS["herd_group"]==hg_name)]
                            sb_any = SOFTBOUNDS[(SOFTBOUNDS["datapoint"]==dp_soft) & (SOFTBOUNDS["breed"]==breed) & 
                                               (SOFTBOUNDS["year"]==2024) & (SOFTBOUNDS["farming_method"]==fm) & 
                                               (SOFTBOUNDS["herd_group"]=="any")]
                            sb_subset = sb_specific if not sb_specific.empty else sb_any
                            if not sb_subset.empty:
                                smin = float(sb_subset["soft_min"].min())
                                smax = float(sb_subset["soft_max"].max())
                                vals = x  # already the dp_col subset values
                                sb_eligible = int(vals.shape[0])
                                sb_below = int(np.sum(vals < smin))
                                sb_above = int(np.sum(vals > smax))
                                total_sb = sb_below + sb_above
                                pct_sb = (total_sb / sb_eligible * 100) if sb_eligible > 0 else 0.0
                                hg_label = f" ({hg_name})" if not sb_specific.empty and hg_name != "All" else ""
                                stat_rows.extend([
                                    {"Metric": f"Softbounds eligible (2024{hg_label})", "Value": sb_eligible},
                                    {"Metric": "Outside softbounds (below)", "Value": sb_below},
                                    {"Metric": "Outside softbounds (above)", "Value": sb_above},
                                    {"Metric": "Outside softbounds (total)", "Value": total_sb},
                                    {"Metric": "Outside softbounds (%)", "Value": f"{pct_sb:.1f}%"},
                                ])
                                
                                # Adjusted bounds statistics (narrows with positive %, broadens with negative %)
                                if show_strict_bounds and strict_pct != 0:
                                    sb_range = smax - smin
                                    adjustment = sb_range * (strict_pct / 100.0)
                                    strict_min = smin + adjustment
                                    strict_max = smax - adjustment
                                    if strict_min < strict_max:
                                        strict_below = int(np.sum(vals < strict_min))
                                        strict_above = int(np.sum(vals > strict_max))
                                        total_strict = strict_below + strict_above
                                        pct_strict = (total_strict / sb_eligible * 100) if sb_eligible > 0 else 0.0
                                        direction_label = f"{'+' if strict_pct > 0 else ''}{strict_pct:.0f}%"
                                        stat_rows.extend([
                                            {"Metric": f"Adjusted bounds min ({direction_label})", "Value": round(strict_min, 3)},
                                            {"Metric": f"Adjusted bounds max ({direction_label})", "Value": round(strict_max, 3)},
                                            {"Metric": "Outside adjusted bounds (below)", "Value": strict_below},
                                            {"Metric": "Outside adjusted bounds (above)", "Value": strict_above},
                                            {"Metric": "Outside adjusted bounds (total)", "Value": total_strict},
                                            {"Metric": "Outside adjusted bounds (%)", "Value": f"{pct_strict:.1f}%"},
                                        ])
                                
                                # Multiplicative bounds statistics
                                if show_multiplicative_bounds and mult_pct > 0:
                                    mult_min = smin * (1 - mult_pct / 100.0)
                                    mult_max = smax * (1 + mult_pct / 100.0)
                                    mult_below = int(np.sum(vals < mult_min))
                                    mult_above = int(np.sum(vals > mult_max))
                                    total_mult = mult_below + mult_above
                                    pct_mult = (total_mult / sb_eligible * 100) if sb_eligible > 0 else 0.0
                                    stat_rows.extend([
                                        {"Metric": f"Multiplicative bounds min (×{1-mult_pct/100:.2f})", "Value": round(mult_min, 3)},
                                        {"Metric": f"Multiplicative bounds max (×{1+mult_pct/100:.2f})", "Value": round(mult_max, 3)},
                                        {"Metric": "Outside multiplicative bounds (below)", "Value": mult_below},
                                        {"Metric": "Outside multiplicative bounds (above)", "Value": mult_above},
                                        {"Metric": "Outside multiplicative bounds (total)", "Value": total_mult},
                                        {"Metric": "Outside multiplicative bounds (%)", "Value": f"{pct_mult:.1f}%"},
                                    ])

                        st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)
                    else:
                        st.info(f"No data for {fm} / {breed} / {hg_name}.")

# Partial r section removed per request
