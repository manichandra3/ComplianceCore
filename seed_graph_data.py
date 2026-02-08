from neo4j import GraphDatabase
import time

class GraphSeeder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def seed_data(self):
        with self.driver.session() as session:
            # 1. Clean existing data (optional, for fresh start)
            session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Cleared existing graph data.")

            # 2. Create Known Mule Accounts (Matches TXN-2026-0002)
            # We create a Mule Account and simulate 6 recent transfers to it to trigger the "Fan-In" rule
            session.run("""
                MERGE (mule:Account {account_id: 'MULE-ACCT-7799', type: 'mule_suspect'})
                
                // Create 6 dummy senders
                FOREACH (i IN range(1, 6) | 
                    MERGE (sender:Account {account_id: 'SENDER-' + i})
                    CREATE (sender)-[:SENDS]->(t:Transaction {
                        transaction_id: 'HIST-TX-' + i, 
                        amount: 1000, 
                        timestamp: datetime() - duration('PT2H')
                    })-[:TO]->(mule)
                )
            """)
            print("✅ Seeded Mule Network (Target: MULE-ACCT-7799)")

            # 3. Create Structuring History (Matches TXN-2026-0003 & TXN-2026-0007)
            # The sender of these transactions needs a history of just-under-$10k transfers
            session.run("""
                MERGE (structurer:Account {account_id: 'ACCT-STRUCTURER-01'})
                
                // Create 3 historical transactions between $9k and $10k in the last 7 days
                FOREACH (i IN range(1, 3) | 
                    CREATE (structurer)-[:SENDS]->(t:Transaction {
                        transaction_id: 'HIST-STRUCT-' + i,
                        amount: 9500,
                        timestamp: datetime() - duration('P' + i + 'D')
                    })
                )
            """)
            print("✅ Seeded Structuring History (Target: ACCT-STRUCTURER-01)")

            # 4. Create Identity Ring / Shared Device (Matches TXN-2026-0004 & TXN-2026-0008)
            # Connect the "New Device" to a known fraudster
            session.run("""
                // Create a known fraudster
                MERGE (fraudster:User {user_id: 'KNOWN_FRAUDSTER_99', status: 'FRAUDSTER'})
                MERGE (fraud_acct:Account {account_id: 'FRAUD-ACCT-99'})
                MERGE (fraudster)-[:OWNS]->(fraud_acct)

                // Create the shared device
                MERGE (bad_device:Device {device_id: 'NEW_DEV-Z1Y2X3'})

                // Link fraudster to this device via a past transaction
                CREATE (fraud_acct)-[:SENDS]->(t:Transaction {
                    transaction_id: 'FRAUD-TX-001',
                    timestamp: datetime() - duration('P30D')
                })-[:HAS_DEVICE]->(bad_device)
            """)
            print("✅ Seeded Identity Ring (Target: NEW_DEV-Z1Y2X3)")

if __name__ == "__main__":
    # Wait for DB to be ready
    time.sleep(5) 
    
    seeder = GraphSeeder("bolt://localhost:7687", "neo4j", "password")
    try:
        seeder.seed_data()
        print("\n🚀 Graph Data Seeding Complete! Rerun your pipeline now.")
    except Exception as e:
        print(f"❌ Error seeding data: {e}")
    finally:
        seeder.close()
