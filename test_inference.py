from app.inference import GPTInference

generator = GPTInference()

print(generator.generate_text(
    prompt="The little rabbit",
    temperature=0.8,
    top_k=40
))

print(generator.generate_text(
    prompt="There was a king",
    temperature=1.0,
    top_k=50
))

print(generator.generate_text(
    prompt="A brave knight",
    temperature=0.7,
    top_k=30
))