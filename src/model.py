# Based on tutorial by NeuralNine: https://youtu.be/JIz-hiRrZ2g, and
# spaCy documentation: https://spacy.io/usage/training#training-data
# adapted for resume skill recognition.

# later to-do:
    # use this dataset: https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences
    # process and turn data into TRAIN_DATA by finding skill in sentence indices
    # train model on this

#maybe run a normal model, cross reference and remove place, people,etc.


# import all necessary modules
import random
import spacy
from spacy.util import minibatch
from spacy.training.example import Example


# Prepare and annotate your training data with entities and labels.
TRAIN_DATA = [
    ("Proficient in Python, Java, and C#.", {"entities": [(14, 20, "SKILL"), (22, 26, "SKILL"), (32, 34, "SKILL")]}),
    ("Expertise with machine learning, deep learning, and AI.", {"entities": [(16, 32, "SKILL"), (34, 47, "SKILL"), (53, 55, "SKILL")]}),
    ("Experienced in JavaScript, React, Angular, and Vue.js.", {"entities": [(14, 24, "SKILL"), (26, 31, "SKILL"), (33, 40, "SKILL"), (46, 52, "SKILL")]}),
    ("Skilled at SQL, NoSQL databases, and data warehousing.", {"entities": [(10, 13, "SKILL"), (15, 20, "SKILL"), (32, 48, "SKILL")]}),
    ("Knowledgeable in cloud platforms such as AWS, Azure, and Google Cloud.", {"entities": [(38, 41, "SKILL"), (43, 48, "SKILL"), (54, 66, "SKILL")]}),
    ("Hands-on experience with Docker, Kubernetes, and Jenkins.", {"entities": [(26, 32, "SKILL"), (34, 44, "SKILL"), (50, 57, "SKILL")]}),
    ("Familiar with Git, SVN, and other version control systems.", {"entities": [(14, 17, "SKILL"), (19, 22, "SKILL")]}),
    ("Proficient in HTML, CSS, and Bootstrap.", {"entities": [(14, 18, "SKILL"), (20, 23, "SKILL"), (29, 38, "SKILL")]}),
    ("Strong background in Agile methodologies and Scrum.", {"entities": [(23, 28, "SKILL"), (41, 46, "SKILL")]}),
    ("Experience using TensorFlow, PyTorch, and Scikit-learn.", {"entities": [(17, 27, "SKILL"), (29, 36, "SKILL"), (42, 55, "SKILL")]}),
    ("Working knowledge of REST APIs and GraphQL.", {"entities": [(20, 24, "SKILL"), (29, 36, "SKILL")]}),
    ("Skilled in data visualization tools like Tableau and Power BI.", {"entities": [(26, 34, "SKILL"), (39, 47, "SKILL")]}),
    ("Experienced in software testing using Selenium and JUnit.", {"entities": [(34, 42, "SKILL"), (47, 52, "SKILL")]}),
    ("Good understanding of networking protocols such as TCP/IP and HTTP.", {"entities": [(41, 46, "SKILL"), (51, 54, "SKILL")]}),
    ("Competent in Microsoft Office Suite, including Excel and PowerPoint.", {"entities": [(15, 32, "SKILL"), (43, 48, "SKILL"), (53, 64, "SKILL")]}),
    ("Proficient with Linux command line tools like Bash and Zsh.", {"entities": [(22, 27, "SKILL"), (45, 49, "SKILL"), (54, 57, "SKILL")]}),
    ("Knowledge of C, C++, and embedded systems development.", {"entities": [(12, 13, "SKILL"), (15, 18, "SKILL"), (24, 42, "SKILL")]}),
    ("Strong communication and leadership skills.", {"entities": [(7, 20, "SKILL"), (25, 35, "SKILL")]}),
    ("Experienced with ERP systems like SAP and Oracle.", {"entities": [(26, 29, "SKILL"), (34, 40, "SKILL")]}),
    ("Familiarity with big data tools such as Hadoop and Spark.", {"entities": [(31, 37, "SKILL"), (42, 47, "SKILL")]}),
    ("Skilled in mobile development with Android and iOS platforms.", {"entities": [(27, 34, "SKILL"), (39, 42, "SKILL")]}),
    ("Working knowledge of DevOps tools including Ansible and Chef.", {"entities": [(28, 35, "SKILL"), (40, 44, "SKILL")]}),
    ("Hands-on experience in UI/UX design using Figma and Adobe XD.", {"entities": [(30, 35, "SKILL"), (40, 49, "SKILL")]}),
    ("Proficient at scripting languages like Perl and Ruby.", {"entities": [(27, 31, "SKILL"), (36, 40, "SKILL")]}),
    ("Experience in blockchain technology and smart contract development.", {"entities": [(16, 26, "SKILL"), (31, 53, "SKILL")]}),
    ("Strong analytical skills with experience in Excel and Python.", {"entities": [(7, 17, "SKILL"), (38, 43, "SKILL"), (48, 54, "SKILL")]}),
    ("Familiar with Agile and Waterfall project management methodologies.", {"entities": [(14, 19, "SKILL"), (24, 33, "SKILL")]}),
    ("Experienced in graphic design using Adobe Photoshop and Illustrator.", {"entities": [(26, 41, "SKILL"), (46, 58, "SKILL")]}),
    ("Good knowledge of virtualization technologies such as VMware and Hyper-V.", {"entities": [(31, 37, "SKILL"), (42, 49, "SKILL")]}),
    ("Proficient in SQL querying and database administration.", {"entities": [(14, 17, "SKILL"), (18, 26, "SKILL")]}),
    ("Experience developing microservices architecture with Spring Boot.", {"entities": [(30, 51, "SKILL")]}),
    ("Skilled in testing automation using Cypress and TestNG.", {"entities": [(28, 35, "SKILL"), (40, 46, "SKILL")]}),
    ("Knowledgeable about container orchestration tools such as OpenShift.", {"entities": [(45, 54, "SKILL")]}),
    ("Familiar with Python libraries like Pandas, NumPy, and Matplotlib.", {"entities": [(13, 19, "SKILL"), (27, 32, "SKILL"), (34, 39, "SKILL"), (45, 55, "SKILL")]}),
    ("Experience working with message brokers such as Kafka and RabbitMQ.", {"entities": [(38, 43, "SKILL"), (48, 56, "SKILL")]}),
    ("Strong background in cybersecurity and ethical hacking.", {"entities": [(23, 36, "SKILL"), (41, 56, "SKILL")]}),
    ("Good knowledge of CRM software like Salesforce and HubSpot.", {"entities": [(23, 33, "SKILL"), (38, 45, "SKILL")]}),
    ("Proficient in RESTful API development and integration.", {"entities": [(14, 28, "SKILL")]}),
    ("Experienced in data engineering with Apache Spark and Hadoop.", {"entities": [(28, 40, "SKILL"), (45, 51, "SKILL")]}),
    ("Familiar with automated deployment using CircleCI and Travis CI.", {"entities": [(33, 41, "SKILL"), (46, 55, "SKILL")]}),
    ("Working knowledge of natural language processing (NLP).", {"entities": [(24, 51, "SKILL")]}),
    ("Strong problem-solving and critical thinking abilities.", {"entities": [(7, 24, "SKILL"), (29, 46, "SKILL")]}),
    ("Experienced in serverless computing using AWS Lambda.", {"entities": [(33, 43, "SKILL")]}),
    ("Familiar with data mining techniques and tools.", {"entities": [(13, 24, "SKILL")]}),
    ("Proficient in scripting with PowerShell and Bash.", {"entities": [(18, 28, "SKILL"), (33, 37, "SKILL")]}),
    ("Hands-on experience with database management systems like MySQL and PostgreSQL.", {"entities": [(51, 56, "SKILL"), (61, 73, "SKILL")]}),
    ("Skilled in front-end technologies including HTML5, CSS3, and JavaScript.", {"entities": [(32, 37, "SKILL"), (39, 43, "SKILL"), (49, 59, "SKILL")]}),
    ("Knowledgeable in software development lifecycle (SDLC) processes.", {"entities": [(16, 49, "SKILL")]}),
    ("Experience with test-driven development (TDD) methodologies.", {"entities": [(18, 49, "SKILL")]}),
    ("Familiarity with cloud-native applications and Kubernetes orchestration.", {"entities": [(27, 49, "SKILL"), (54, 64, "SKILL")]}),
    ("Proficient in Rust, Go, and Elixir.", {"entities": [(14, 18, "SKILL"), (20, 22, "SKILL"), (28, 34, "SKILL")]}),
    ("Experienced in data pipelines using Apache Beam and NiFi.", {"entities": [(21, 35, "SKILL"), (42, 53, "SKILL")]}),
    ("Knowledgeable in containerization with Docker and Podman.", {"entities": [(18, 34, "SKILL"), (40, 46, "SKILL"), (51, 57, "SKILL")]}),
    ("Hands-on experience in test automation using Robot Framework.", {"entities": [(26, 42, "SKILL"), (49, 65, "SKILL")]}),
    ("Strong understanding of CI/CD pipelines and GitOps.", {"entities": [(26, 41, "SKILL"), (46, 52, "SKILL")]}),
    ("Skilled in data cleaning and preprocessing with Pandas.", {"entities": [(14, 27, "SKILL"), (32, 46, "SKILL"), (52, 58, "SKILL")]}),
    ("Experienced in backend frameworks such as Express and Django.", {"entities": [(21, 40, "SKILL"), (50, 57, "SKILL"), (62, 68, "SKILL")]}),
    ("Proficient in infrastructure as code tools like Terraform and Pulumi.", {"entities": [(14, 38, "SKILL"), (49, 58, "SKILL"), (63, 69, "SKILL")]}),
    ("Working knowledge of real-time data streaming with Apache Flink.", {"entities": [(20, 45, "SKILL"), (51, 64, "SKILL")]}),
    ("Strong skills in performance testing using LoadRunner.", {"entities": [(17, 37, "SKILL"), (44, 54, "SKILL")]}),
    ("Familiar with security tools like Nessus and Metasploit.", {"entities": [(21, 27, "SKILL"), (32, 42, "SKILL")]}),
    ("Good experience with version control using Mercurial and GitLab.", {"entities": [(38, 47, "SKILL"), (52, 60, "SKILL")]}),
    ("Skilled at writing shell scripts in Bash and Fish.", {"entities": [(25, 31, "SKILL"), (36, 40, "SKILL"), (45, 49, "SKILL")]}),
    ("Expert in scientific computing with SciPy and SymPy.", {"entities": [(29, 34, "SKILL"), (39, 44, "SKILL")]}),
    ("Knowledgeable in ITIL framework and IT service management.", {"entities": [(18, 22, "SKILL"), (35, 58, "SKILL")]}),
    ("Hands-on experience using development tools like Xcode and Android Studio.", {"entities": [(38, 43, "SKILL"), (48, 63, "SKILL")]}),
    ("Proficient with big data platforms including Hive and Presto.", {"entities": [(24, 44, "SKILL"), (55, 59, "SKILL"), (64, 70, "SKILL")]}),
    ("Experienced in working with ETL processes using Talend.", {"entities": [(28, 42, "SKILL"), (50, 56, "SKILL")]}),
    ("Familiarity with data science libraries like Seaborn and Statsmodels.", {"entities": [(35, 42, "SKILL"), (47, 59, "SKILL")]}),
    ("Knowledgeable in software container technologies like LXC.", {"entities": [(18, 53, "SKILL"), (59, 62, "SKILL")]}),
    ("Proficient in CAD tools such as AutoCAD and SolidWorks.", {"entities": [(14, 17, "SKILL"), (32, 39, "SKILL"), (44, 55, "SKILL")]}),
    ("Working with distributed systems and eventual consistency.", {"entities": [(13, 33, "SKILL"), (38, 59, "SKILL")]}),
    ("Skilled in using monitoring tools like Prometheus and Grafana.", {"entities": [(24, 41, "SKILL"), (47, 57, "SKILL")]}),
    ("Experienced in frontend frameworks like Ember.js and Svelte.", {"entities": [(21, 42, "SKILL"), (48, 56, "SKILL")]}),
    ("Strong background in NoSQL solutions such as Couchbase and Cassandra.", {"entities": [(23, 39, "SKILL"), (49, 58, "SKILL"), (63, 72, "SKILL")]}),
    ("Familiar with operating systems including Unix and FreeBSD.", {"entities": [(14, 32, "SKILL"), (43, 47, "SKILL"), (52, 59, "SKILL")]}),
    ("Proficient in graph databases like Neo4j and ArangoDB.", {"entities": [(14, 30, "SKILL"), (36, 40, "SKILL"), (45, 53, "SKILL")]}),
    ("Hands-on experience with Business Intelligence tools like QlikView.", {"entities": [(26, 53, "SKILL"), (59, 67, "SKILL")]}),
    ("Skilled in hardware description languages such as VHDL and Verilog.", {"entities": [(14, 49, "SKILL"), (59, 63, "SKILL"), (68, 75, "SKILL")]}),
    ("Experienced in analytics platforms such as Looker and Domo.", {"entities": [(21, 42, "SKILL"), (52, 58, "SKILL"), (63, 67, "SKILL")]}),
    ("Knowledge of mobile frameworks like Flutter and Ionic.", {"entities": [(12, 30, "SKILL"), (36, 43, "SKILL"), (48, 53, "SKILL")]}),
    ("Familiar with documentation tools like Confluence and Notion.", {"entities": [(14, 37, "SKILL"), (43, 53, "SKILL"), (58, 64, "SKILL")]}),
    ("Proficient at CRM and ERP integrations.", {"entities": [(14, 17, "SKILL"), (22, 25, "SKILL"), (26, 40, "SKILL")]}),
    ("Strong understanding of AI/ML workflows and model deployment.", {"entities": [(26, 43, "SKILL"), (48, 65, "SKILL")]}),
    ("Experienced with logging tools like ELK Stack and Fluentd.", {"entities": [(25, 37, "SKILL"), (43, 52, "SKILL")]}),
    ("Working with quantum computing concepts and Qiskit framework.", {"entities": [(13, 32, "SKILL"), (37, 53, "SKILL")]}),
    ("Proficient in e-commerce platforms like Magento and Shopify.", {"entities": [(14, 34, "SKILL"), (40, 47, "SKILL"), (52, 60, "SKILL")]}),
    ("Familiarity with scientific visualization using ParaView and VisIt.", {"entities": [(22, 51, "SKILL"), (58, 66, "SKILL"), (71, 76, "SKILL")]}),
    ("Hands-on experience in big data processing with Dask.", {"entities": [(26, 47, "SKILL"), (53, 57, "SKILL")]}),
    ("Strong problem-solving with decision trees and random forests.", {"entities": [(7, 24, "SKILL"), (30, 45, "SKILL"), (50, 64, "SKILL")]}),
    ("Knowledgeable in multimedia tools such as Final Cut Pro and DaVinci Resolve.", {"entities": [(18, 34, "SKILL"), (45, 58, "SKILL"), (63, 79, "SKILL")]}),
    ("Experienced in financial software like QuickBooks and Xero.", {"entities": [(21, 39, "SKILL"), (45, 56, "SKILL"), (61, 65, "SKILL")]}),
    ("Familiar with search engine technologies such as Solr and Elasticsearch.", {"entities": [(14, 44, "SKILL"), (50, 54, "SKILL"), (59, 72, "SKILL")]}),
    ("Working knowledge of API testing tools like Postman and SoapUI.", {"entities": [(20, 41, "SKILL"), (47, 54, "SKILL"), (59, 65, "SKILL")]}),
    ("Skilled in edge computing and IoT development.", {"entities": [(14, 29, "SKILL"), (34, 49, "SKILL")]}),
    ("Proficient in robotics software such as ROS and V-REP.", {"entities": [(14, 32, "SKILL"), (43, 46, "SKILL"), (51, 56, "SKILL")]}),
    ("Experienced in technical writing and API documentation.", {"entities": [(21, 38, "SKILL"), (43, 60, "SKILL")]}),
    ("Familiarity with biometric authentication systems and facial recognition.", {"entities": [(18, 50, "SKILL"), (55, 73, "SKILL")]}),
]


# Load a pretrained SpaCy model or create a blank one.
nlp = spacy.load("en_core_web_trf")

# Add or get the Named Entity Recognizer (NER) pipeline component.
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

# Add your custom entity labels to the NER component.
for text, annotations in TRAIN_DATA:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2]) 

# Disable other pipeline components during training for better performance.
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ["ner", "transformer"]]
with nlp.disable_pipes(*other_pipes):
    # Initialize the optimizer to start training.
    optimizer = nlp.resume_training()

    # Train your model over multiple epochs, updating it with your training data.
    epochs = 50
    for epoch in range(epochs):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=3)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f'Epoch trained: {epoch+1}, Losses: {losses}')

# Save the trained model to disk for future use.
nlp.to_disk('resume_model')