import pandas as pd

# Sample data for concept and description pairs
data = {
    'City': [
        'Bangalore',
        'Delhi',
        'Hyderabad',
        'Chennai',
        'Kolkata',
        'Kochi',
        'Tirupati',
        'Mumbai',
        'Goa',
        'Lucknow'
    ],
    'Movie': [
        'Animal',
        'Aquaman 2',
        'Devil: The British Secret Agent',
        'Dunki',
        'Hi Nanna',
        'Neru',
        'Salaar',
        'Devil: The British Secret Agent',  # Repeated for variety
        'Animal',  # Repeated for variety
        'Aquaman 2'  # Repeated for variety
    ],
    'Theatre': [
        'Cinepolis: Nexus Shantiniketan, Bengaluru',
        'Gopalan Miniplex: Signature Mall, Old Madras Road',
        'PVR: Nexus (Formerly Forum), Koramangala',
        'PVR: VR Bengaluru, Whitefield Road',
        'INOX: Nehru Place, New Delhi',
        'PVR: DLF Mall of India, Noida',
        'Cinepolis: Inorbit Mall, Hyderabad',
        'PVR: Forum Mall, Vadapalani, Chennai',
        'INOX: Quest Mall, Kolkata',
        'Cinepolis: Nexus Shantiniketan, Bengaluru'  # Repeated for variety
    ],
    'Show_Timings': [
        '10:00AM', '2:00PM', '6:00PM', '9:00PM',
        '11:00AM', '3:00PM', '7:00PM', '10:00PM',
        '1:00PM', '5:00PM', '8:00PM', '11:00PM',
        '12:00PM', '4:00PM', '7:30PM', '10:30PM',
        '2:30PM', '6:30PM', '9:30PM', '12:30AM'
    ]
}

# Ensure all columns have the same length
max_length = max(len(data['City']), len(data['Movie']), len(data['Theatre']), len(data['Show_Timings']))
for key in data:
    data[key] = (data[key] + [None] * (max_length - len(data[key])))[:max_length]

# Create DataFrame
df = pd.DataFrame(data)

# Add 'text' column
df['text'] = (
    f"###Human:\n"
    f"1. User: Hi\n"
    f"2. Bot: Hello! How can I assist you today?\n"
    f"3. User: I want to book movie tickets.\n"
    f"4. Bot: Sure! Let me show you the available cities: {df['City']}\n"
    f"5. User: I choose Bangalore\n"
    f"6. Bot: Great choice! Here are the movies in Bangalore: {df['Movie']}\n"
    f"7. User: I want to watch 'Aquaman 2'\n"
    f"8. Bot: Excellent! Now, please choose a theatre: {df['Theatre']}\n"
    f"9. User: I select 'PVR: VR Bengaluru, Whitefield Road'\n"
    f"10. Bot: Perfect! Now, let me know the preferred show timing: {df['Show_Timings']}\n"
    f"11. User: I choose the show at 7:00 PM\n"
    f"12. Bot: How many tickets would you like? (Choose from 1-6)\n"
    f"13. User: I want 2 tickets\n"
    f"14. Bot: Great! Here is your confirmation: [Details]\n"
    f"15. User: Thank you!\n"
    f"16. Bot: You're welcome! Enjoy the movie.\n"
)

# Write DataFrame to CSV file
df.to_csv('booking_prompts.csv', index=False)

print("CSV file created successfully.")
