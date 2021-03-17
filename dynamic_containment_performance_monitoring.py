# Please note, this file is for illustrative purposes only and should not be relied on, or treated as a substitute for the Service Terms.

import pandas
import numpy as np
from math import floor, ceil, inf
from datetime import datetime

## Apply separate upwards and downwards ramp limits
def ramp_limit(t, v_in, up, down):
    dt = t[2] - t[1]
    up_dt = up * dt
    down_dt = down * dt
    
    v = v_in.copy()
    
    for i in range(1, len(v)):
        change = v[i] - v[i-1]
        
        if change > up_dt:
            v[i] = v[i-1] + up_dt
        elif change < -down_dt:
            v[i] = v[i-1] - down_dt
    
    return v

# Apply the response curve to a vector of frequencies
def response_curve(f, freq = np.array([49.5,49.8,49.985]), frac=np.array([1, 0.05, 0])):
    return np.interp(f, freq, frac)

# Get the minimum frequency between min_lag and max_lag ago
def low_freq_window(t, f, min_lag = 0.2, max_lag = 0.55):
    # This version is only looking data quantized to 0.05 s from the hour so it doesn't use t for much
    dt = t[1] - t[0]
    
    min_step = round(min_lag / dt)
    max_step = round(max_lag / dt)
    
    # Calculate lagged rolling minimum
    new_f = f.rolling(max_step-min_step+1, min_periods=1).min().shift(min_step, fill_value=f[0])
    
    return new_f

# Get the maximum frequency between min_lag and max_lag ago
def high_freq_window(t, f, min_lag = 0.2, max_lag = 0.55):
    dt = t[1] - t[0]
    
    min_step = round(min_lag / dt)
    max_step = round(max_lag / dt)
    
    # Calculate lagged rolling maximum
    new_f = f.rolling(max_step-min_step+1, min_periods=1).max().shift(min_step, fill_value=f[0])
    
    return new_f

# Calculate lower performance bound
def lower_bound(t, f, min_lag = 0.2, max_lag = 0.55, min_ramp_rate = 2, max_ramp_rate = 4):
    # Get high frequencies
    new_f = high_freq_window(t, f, min_lag, max_lag)
    # Calculate response curve
    no_ramp = response_curve(new_f)
    #Apply ramp limit
    ramp = ramp_limit(t, no_ramp, min_ramp_rate, max_ramp_rate)
    
    return ramp

# Calculate upper bound
def upper_bound(t, f, min_lag = 0.2, max_lag = 0.55, min_ramp_rate = 2, max_ramp_rate = 4):
    # Get low frequencies
    new_f = low_freq_window(t, f, min_lag, max_lag)
    # Calculate response curve
    no_ramp = response_curve(new_f)
    #Apply ramp limit
    ramp = ramp_limit(t, no_ramp, max_ramp_rate, min_ramp_rate)
    
    return ramp


# Read in input file ####

input_file = "Data/sample.csv"
input_df = pandas.read_csv(input_file)

# Convert to datetime, then to seconds from the first datapoint
ts = input_df.t.transform(lambda t: datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ'))
t = ts.transform(lambda ti : (ti - ts[0]).total_seconds())

# Get other relevant variables
f = input_df.f_hz
response = input_df.p_mw - input_df.baseline_mw
availability = input_df.availability

# Contracted times and MW are read in from a file in the actual tool
contracted = pandas.Series([True for x in range(len(f))])
MW = 50


# Bounds calculation ####

lb = lower_bound(t,f) * MW
ub = upper_bound(t,f) * MW

# For the first few timesteps after the start of the data or a settlement period gap, set bounds to max as we don't have any lagged frequency
# (The tool receives daily data per unit rather than the individual hourly files)
gap_mask = t.diff() > 29 * 60
fill_mask = gap_mask.rolling(21,min_periods=1).apply(lambda x : x.any(), raw=True).astype(bool) # Extend mask 1 second forward for the 0.5 seconds lag and 0.5 seconds ramp time
lb[fill_mask] = 0
ub[fill_mask] = MW

# Ignore times with unavailability and times not contracted
ua_mask = availability == 0 | ~contracted
lb[ua_mask] = -inf
ub[ua_mask] = inf


# Calculate error ####

## Distance from upper/lower bounds
def bound_error(v, lower, upper):
  error = np.where(v < lower, lower - v, np.where(v > upper, v - upper, 0))
  return error

# Vectors of error values for each time
error = bound_error(response, lb, ub)
proportional_error = error / MW
rolling_error = pandas.Series(proportional_error).rolling(4, min_periods=1).min()

# Final error value used in settlements calculation
result = max(rolling_error)
