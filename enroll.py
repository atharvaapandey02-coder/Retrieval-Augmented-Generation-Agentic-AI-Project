from rag_engine import enroll_person

print("\n Enrolling people...\n")

base = "static/enroll"

enroll_person("akshay", f"{base}/akshay.jpg")
enroll_person("alia", f"{base}/alia.jpg")
enroll_person("deepika", f"{base}/deepika.jpg")
enroll_person("kohli", f"{base}/kohli.jpg")
enroll_person("anirkhan", f"{base}/anirkhan.jpg")
enroll_person("vaibhav", f"{base}/vaibhav.jpg")
enroll_person("shraddha", f"{base}/shraddha.jpg")
enroll_person("rohitsharma", f"{base}/rohitsharma.jpg")
enroll_person("ranveer", f"{base}/ranveer.jpg")
enroll_person("kartikaaryan", f"{base}/kartikaaryan.jpg")
enroll_person("bachachan", f"{base}/bachachan.jpg")
print("\n Enrollment complete.\n")
