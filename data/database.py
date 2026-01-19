"""
Mock database for sample documents and testing.
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class MockDatabase:
    """Mock database for storing sample documents."""
    
    def __init__(self, db_path: str = "data/sample_documents.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with sample documents."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        ''')
        
        # Insert sample categories
        categories = [
            ("Technology", "Articles about technology and innovation"),
            ("Science", "Scientific research and discoveries"),
            ("Business", "Business news and analysis"),
            ("Health", "Health and medical information"),
            ("Education", "Educational content and research")
        ]
        
        cursor.executemany(
            'INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)',
            categories
        )
        
        # Insert sample documents
        sample_documents = self._get_sample_documents()
        cursor.executemany(
            'INSERT OR IGNORE INTO documents (title, content, category, source) VALUES (?, ?, ?, ?)',
            sample_documents
        )
        
        conn.commit()
        conn.close()
    
    def _get_sample_documents(self) -> List[tuple]:
        """Get sample documents for the database."""
        return [
            (
                "Artificial Intelligence Revolution",
                "Artificial intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI technologies are enabling unprecedented levels of automation and decision-making capabilities. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy. Deep learning, a subset of machine learning, has revolutionized fields like computer vision and natural language processing. Companies are investing billions in AI research and development, recognizing its potential to drive innovation and competitive advantage. However, the rapid advancement of AI also raises important questions about ethics, privacy, and the future of work. As AI systems become more sophisticated, society must grapple with issues of bias, transparency, and accountability. The next decade will likely see AI become even more integrated into our daily lives, from autonomous vehicles to personalized medicine.",
                "Technology",
                "AI Research Institute"
            ),
            (
                "Climate Change and Renewable Energy",
                "Climate change represents one of the most pressing challenges of our time. Rising global temperatures, melting ice caps, and extreme weather events are clear indicators of the urgent need for action. Renewable energy sources such as solar, wind, and hydroelectric power offer promising solutions to reduce greenhouse gas emissions. Solar panel technology has become increasingly efficient and cost-effective, making it accessible to more households and businesses. Wind farms are generating significant amounts of clean electricity in many regions. Governments worldwide are implementing policies to accelerate the transition to renewable energy. The Paris Agreement has brought nations together to commit to limiting global temperature rise. However, challenges remain in terms of energy storage, grid infrastructure, and the need for continued innovation. Investment in renewable energy technologies continues to grow, driven by both environmental concerns and economic opportunities.",
                "Science",
                "Environmental Research Center"
            ),
            (
                "Remote Work and Digital Transformation",
                "The COVID-19 pandemic accelerated the adoption of remote work practices across industries. Companies that previously resisted telecommuting were forced to adapt quickly to maintain operations. Digital transformation became not just a strategic initiative but a necessity for survival. Video conferencing platforms saw unprecedented growth as teams learned to collaborate virtually. Cloud computing services became essential for enabling remote access to business applications and data. Cybersecurity concerns increased as organizations expanded their digital footprints. Employee productivity and work-life balance became key considerations for remote work policies. Many companies are now adopting hybrid work models that combine remote and in-office work. The shift has also impacted commercial real estate markets and urban planning. Remote work has opened up new opportunities for talent acquisition beyond geographic boundaries.",
                "Business",
                "Business Technology Review"
            ),
            (
                "Mental Health in the Digital Age",
                "The digital age has brought both opportunities and challenges for mental health. Social media platforms can provide valuable connections and support networks, but they can also contribute to anxiety, depression, and social comparison. The constant connectivity enabled by smartphones and digital devices has blurred the boundaries between work and personal life. Digital wellness has become an important consideration for individuals and organizations. Mindfulness apps and online therapy platforms have made mental health resources more accessible. However, the quality and effectiveness of digital mental health interventions vary widely. Research is ongoing to understand the long-term effects of digital technology on mental health. Some studies suggest that excessive screen time can negatively impact sleep quality and attention spans. Organizations are increasingly recognizing the importance of employee mental health and implementing wellness programs. The future of mental health care may involve more personalized, technology-enabled approaches.",
                "Health",
                "Digital Health Institute"
            ),
            (
                "Space Exploration and Mars Missions",
                "Space exploration has entered a new era with ambitious missions to Mars and beyond. NASA's Perseverance rover continues to explore the Martian surface, searching for signs of past life and collecting samples for future return to Earth. Private companies like SpaceX are developing reusable rocket technology that could dramatically reduce the cost of space travel. The Artemis program aims to return humans to the Moon and establish a sustainable presence there. Mars colonization remains a long-term goal for space agencies and private enterprises. Advances in propulsion technology, life support systems, and radiation protection are making longer space missions more feasible. International cooperation in space exploration has strengthened relationships between nations. The search for extraterrestrial life continues through various missions and telescope observations. Space tourism is becoming a reality for wealthy individuals. The economic potential of space resources, such as asteroid mining, is being explored by commercial ventures.",
                "Science",
                "Space Research Foundation"
            ),
            (
                "Cryptocurrency and Blockchain Technology",
                "Cryptocurrency has evolved from a niche technology to a mainstream financial instrument. Bitcoin and other digital currencies have gained acceptance from institutional investors and corporations. Blockchain technology, the underlying system for cryptocurrencies, has applications far beyond digital money. Smart contracts enable automated execution of agreements without intermediaries. Decentralized finance (DeFi) platforms are creating new ways to access financial services. Non-fungible tokens (NFTs) have created new markets for digital art and collectibles. Central banks are exploring the development of digital currencies. However, cryptocurrency markets remain highly volatile and speculative. Regulatory frameworks are still being developed to address concerns about security, fraud, and environmental impact. The energy consumption of cryptocurrency mining has raised environmental concerns. Despite challenges, blockchain technology continues to show promise for improving transparency and efficiency in various industries.",
                "Technology",
                "Blockchain Research Lab"
            ),
            (
                "Sustainable Agriculture and Food Security",
                "Sustainable agriculture practices are essential for ensuring food security while protecting the environment. Precision farming techniques use data analytics and sensors to optimize crop yields and reduce resource consumption. Vertical farming and hydroponic systems are enabling food production in urban environments. Organic farming methods are gaining popularity among consumers concerned about health and environmental impact. Climate-smart agriculture adapts farming practices to changing weather patterns. Food waste reduction initiatives are addressing the significant amount of food lost throughout the supply chain. Plant-based alternatives to meat are becoming more popular and accessible. Agricultural technology startups are developing innovative solutions for farmers. International cooperation is needed to address global food security challenges. The role of small-scale farmers in sustainable agriculture is being recognized and supported. Research into drought-resistant crops and sustainable farming methods continues to advance.",
                "Science",
                "Agricultural Research Institute"
            ),
            (
                "Electric Vehicles and Transportation",
                "Electric vehicles (EVs) are rapidly gaining market share as consumers and governments embrace cleaner transportation options. Battery technology improvements have extended driving ranges and reduced charging times. Charging infrastructure is expanding to support the growing number of electric vehicles on the road. Major automakers are investing heavily in electric vehicle development and production. Government incentives and regulations are encouraging the adoption of electric vehicles. The environmental benefits of electric vehicles depend on the source of electricity generation. Autonomous vehicle technology is being integrated with electric powertrains. The transition to electric vehicles is creating new opportunities in the automotive industry. Challenges remain in terms of battery recycling and the environmental impact of battery production. Public transportation systems are also electrifying their fleets. The future of transportation may involve a combination of electric, autonomous, and shared mobility solutions.",
                "Technology",
                "Transportation Innovation Center"
            )
        ]
    
    def get_documents(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Retrieve documents from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                'SELECT id, title, content, category, source, created_at FROM documents WHERE category = ? ORDER BY created_at DESC LIMIT ?',
                (category, limit)
            )
        else:
            cursor.execute(
                'SELECT id, title, content, category, source, created_at FROM documents ORDER BY created_at DESC LIMIT ?',
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'source': row[4],
                'created_at': row[5]
            })
        
        return documents
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Retrieve a specific document by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, title, content, category, source, created_at FROM documents WHERE id = ?',
            (doc_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'source': row[4],
                'created_at': row[5]
            }
        
        return None
    
    def get_categories(self) -> List[Dict]:
        """Retrieve all categories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, description FROM categories ORDER BY name')
        rows = cursor.fetchall()
        conn.close()
        
        categories = []
        for row in rows:
            categories.append({
                'id': row[0],
                'name': row[1],
                'description': row[2]
            })
        
        return categories
    
    def add_document(self, title: str, content: str, category: str, source: str = "User Input") -> int:
        """Add a new document to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO documents (title, content, category, source) VALUES (?, ?, ?, ?)',
            (title, content, category, source)
        )
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return doc_id
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents by title or content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        search_query = f"%{query}%"
        cursor.execute(
            'SELECT id, title, content, category, source, created_at FROM documents WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?',
            (search_query, search_query, limit)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'source': row[4],
                'created_at': row[5]
            })
        
        return documents
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the documents in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        # Documents by category
        cursor.execute('SELECT category, COUNT(*) FROM documents GROUP BY category')
        category_counts = dict(cursor.fetchall())
        
        # Average document length
        cursor.execute('SELECT AVG(LENGTH(content)) FROM documents')
        avg_length = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': total_docs,
            'documents_by_category': category_counts,
            'average_length': avg_length
        }
