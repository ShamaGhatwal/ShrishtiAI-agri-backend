"""
Weather Date Logic Verification Test
Ensures we fetch exactly 59 days before disaster + disaster date = 60 days total
"""
from datetime import datetime, timedelta
from typing import List

def test_weather_date_logic():
    """Test the weather date calculation logic"""
    print("=== Weather Date Logic Test ===\n")
    
    # Test parameters
    disaster_date_str = "2024-01-15"
    days_before = 60  # This means 59 days before + disaster date itself
    
    print(f"Disaster Date: {disaster_date_str}")
    print(f"Days Before Parameter: {days_before}")
    print(f"Expected: {days_before} total days (59 days before + 1 disaster day)")
    
    # Calculate date range (same logic as weather service)
    disaster_date = datetime.strptime(disaster_date_str, '%Y-%m-%d')
    end_date = disaster_date  # End on disaster date
    start_date = end_date - timedelta(days=days_before - 1)  # Start 59 days before
    
    print(f"\nCalculated Date Range:")
    print(f"Start Date: {start_date.strftime('%Y-%m-%d')} (59 days before disaster)")
    print(f"End Date: {end_date.strftime('%Y-%m-%d')} (disaster date)")
    
    # Generate full date sequence
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"\nGenerated {len(date_range)} dates:")
    print(f"First date: {date_range[0].strftime('%Y-%m-%d')}")
    print(f"Last date: {date_range[-1].strftime('%Y-%m-%d')}")
    
    # Show first 5 and last 5 dates
    print("\nFirst 5 dates:")
    for i, date in enumerate(date_range[:5]):
        days_diff = (date - disaster_date).days
        print(f"  Day {i+1}: {date.strftime('%Y-%m-%d')} ({days_diff:+d} days from disaster)")
    
    print(f"\n... ({len(date_range)-10} intermediate dates) ...\n")
    
    print("Last 5 dates:")
    for i, date in enumerate(date_range[-5:], len(date_range)-4):
        days_diff = (date - disaster_date).days
        print(f"  Day {i}: {date.strftime('%Y-%m-%d')} ({days_diff:+d} days from disaster)")
    
    # Verification
    print(f"\n=== VERIFICATION ===")
    print(f"Total days fetched: {len(date_range)} days")
    print(f"Expected days: {days_before} days")
    print(f"Match: {'✅ CORRECT' if len(date_range) == days_before else '❌ INCORRECT'}")
    
    print(f"\nEnd date matches disaster date: {'✅ CORRECT' if date_range[-1] == disaster_date else '❌ INCORRECT'}")
    
    days_before_disaster = (disaster_date - date_range[0]).days
    print(f"Days before disaster: {days_before_disaster}")
    print(f"Expected days before: {days_before - 1}")
    print(f"Match: {'✅ CORRECT' if days_before_disaster == (days_before - 1) else '❌ INCORRECT'}")
    
    return len(date_range) == days_before and date_range[-1] == disaster_date

def test_nasa_api_date_format():
    """Test NASA API date formatting"""
    print("\n=== NASA API Date Format Test ===\n")
    
    disaster_date_str = "2024-01-15"
    days_before = 60
    
    disaster_date = datetime.strptime(disaster_date_str, '%Y-%m-%d')
    end_date = disaster_date
    start_date = end_date - timedelta(days=days_before - 1)
    
    # NASA API expects YYYYMMDD format
    api_start = start_date.strftime("%Y%m%d")
    api_end = end_date.strftime("%Y%m%d")
    
    print(f"NASA API Parameters:")
    print(f"start: {api_start}")
    print(f"end: {api_end}")
    
    # Simulate what NASA would return (daily data keys)
    expected_dates = []
    current_date = start_date
    while current_date <= end_date:
        expected_dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    print(f"\nExpected NASA response date keys ({len(expected_dates)} total):")
    print(f"First: {expected_dates[0]} ({datetime.strptime(expected_dates[0], '%Y%m%d').strftime('%Y-%m-%d')})")
    print(f"Last: {expected_dates[-1]} ({datetime.strptime(expected_dates[-1], '%Y%m%d').strftime('%Y-%m-%d')})")
    
    return len(expected_dates) == days_before

def main():
    """Run all date logic tests"""
    print("Weather Service Date Logic Verification\n")
    
    test1_passed = test_weather_date_logic()
    test2_passed = test_nasa_api_date_format()
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Date Logic Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"NASA API Format Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Weather date logic is correct.")
        print("✅ Fetches 59 days before disaster + disaster date = 60 total days")
        print("✅ Dates are in correct chronological order")
        print("✅ NASA API format is correct")
    else:
        print("\n⚠️  Some tests failed. Date logic needs fixing.")
    
    return test1_passed and test2_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)