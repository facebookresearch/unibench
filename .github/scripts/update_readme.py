#!/usr/bin/env python3

import re
import os

def get_latest_update():
    """Extract the latest update section from UPDATES.md"""
    with open("assets/UPDATES.md", "r") as file:
        content = file.read()
    
    # Find all update sections (starting with ## and followed by date/version)
    update_sections = re.findall(r'(## .+?(?=## |\Z))', content, re.DOTALL)
    
    if not update_sections:
        return None
    
    # Return the first (latest) update section
    latest_update = update_sections[0].strip()
    
    # Convert the section to a list of lines for easier processing
    update_lines = latest_update.split('\n')
    
    # Get the heading (first line) and change ## to ####
    heading = update_lines[0].replace('##', '####')
    
    # Get the content (everything after the heading)
    content_lines = update_lines[1:]
    content = '\n'.join([line.strip() for line in content_lines if line.strip()])
    
    return {
        'heading': heading,
        'content': content
    }

def update_readme(latest_update):
    """Update the News and Updates section in README.md"""
    with open("README.md", "r") as file:
        readme_content = file.read()
    
    # Define patterns to find the start of News section and the next section after it
    news_section_start = r'## News and Updates\s+For the latest news and updates, see the snippet below.\s+'
    news_start_match = re.search(news_section_start, readme_content)
    
    if not news_start_match:
        print("News and Updates section not found in README.md")
        return False
    
    # Find the next section header after News and Updates
    next_section_pattern = r'(?m)^## [^#]'
    next_sections = list(re.finditer(next_section_pattern, readme_content))
    
    # Find the position of the section after News and Updates
    news_section_pos = news_start_match.start()
    next_section_pos = None
    
    for section in next_sections:
        if section.start() > news_section_pos:
            next_section_pos = section.start()
            break
    
    if not next_section_pos:
        print("Could not find the section after News and Updates")
        return False
    
    # Extract the parts before News section, the News section intro, and after the News section
    before_news = readme_content[:news_start_match.end()]
    after_news = readme_content[next_section_pos:]
    
    # Create the new README content with a single update section
    new_update_section = f"{latest_update['heading']}\n{latest_update['content']}\n  \nFor full details, refer to the [UPDATES.md](./assets/UPDATES.md) file.\n\n"
    updated_readme = before_news + new_update_section + after_news
    
    # Write the updated content back to README.md
    with open("README.md", "w") as file:
        file.write(updated_readme)
    
    return True

def main():
    latest_update = get_latest_update()
    if latest_update:
        success = update_readme(latest_update)
        if success:
            print(f"Successfully updated README.md with latest update: {latest_update['heading']}")
        else:
            print("Failed to update README.md")
    else:
        print("No updates found in UPDATES.md")

if __name__ == "__main__":
    main()
