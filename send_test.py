import resend, os

resend.api_key = os.environ.get("RESEND_API_KEY")

r = resend.Emails.send({
    "from": "onboarding@resend.dev",
    "to": os.environ.get("DIGEST_TO"),
    "subject": "Test from ScholarPulse",
    "text": "If you see this, email is working!"
})
print(r)
