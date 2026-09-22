import random
import numpy as np

random.seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1. Generate text dataset
# --------------------------------------------------

def make_example():
    even = random.randrange(0, 100, 2)
    odd = random.randrange(1, 100, 2)

    if random.random() < 0.5:
        a, b = even, odd
    else:
        a, b = odd, even

    question = f"Which number is odd: {a} or {b}?"
    answer = f"{odd} is odd. {even} is even."

    # Simple numerical representation of the text input
    features = np.array([
        a / 100,
        b / 100,
        a % 2,
        b % 2
    ])

    # 0 = first number is odd
    # 1 = second number is odd
    label = 0 if a % 2 else 1

    return question, answer, features, label


dataset = [make_example() for _ in range(1000)]

train = dataset[:800]
test = dataset[800:]

X_train = np.stack([x[2] for x in train])
y_train = np.array([x[3] for x in train])

X_test = np.stack([x[2] for x in test])
y_test = np.array([x[3] for x in test])


# --------------------------------------------------
# 2. Softmax
# --------------------------------------------------

def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# --------------------------------------------------
# 3. Teacher model
# --------------------------------------------------

def teacher(X):
    first_odd = 5 * (X[:, 2] - X[:, 3])
    second_odd = 5 * (X[:, 3] - X[:, 2])

    return np.column_stack([first_odd, second_odd])


teacher_train_logits = teacher(X_train)
teacher_test_logits = teacher(X_test)


# --------------------------------------------------
# 4. Train student
# --------------------------------------------------

def train_student(use_kd=False, epochs=300, lr=0.25,
                  temperature=3.0, alpha=0.65):

    W = np.random.normal(0, 0.05, (4, 2))
    b = np.zeros(2)

    labels = np.eye(2)[y_train]

    for epoch in range(epochs):

        student_logits = X_train @ W + b
        student_probs = softmax(student_logits)

        # Normal supervised loss gradient
        ce_gradient = (student_probs - labels) / len(X_train)

        if use_kd:

            teacher_probs = softmax(
                teacher_train_logits / temperature
            )

            student_soft = softmax(
                student_logits / temperature
            )

            kd_gradient = (
                student_soft - teacher_probs
            ) * temperature / len(X_train)

            gradient = (
                (1 - alpha) * ce_gradient
                + alpha * kd_gradient
            )

        else:
            gradient = ce_gradient

        W -= lr * (X_train.T @ gradient)
        b -= lr * gradient.sum(axis=0)

    return W, b


# --------------------------------------------------
# 5. Train normal student and KD student
# --------------------------------------------------

normal_W, normal_b = train_student(use_kd=False)

kd_W, kd_b = train_student(use_kd=True)


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

def accuracy(W, b):

    predictions = (X_test @ W + b).argmax(axis=1)

    return (predictions == y_test).mean()


teacher_accuracy = (
    teacher_test_logits.argmax(axis=1) == y_test
).mean()

normal_accuracy = accuracy(normal_W, normal_b)
kd_accuracy = accuracy(kd_W, kd_b)


print("\nKNOWLEDGE DISTILLATION — ODD/EVEN DEMO")
print("---------------------------------------")

print(f"Training examples: {len(train)}")
print(f"Testing examples:  {len(test)}")

print()
print(f"Teacher accuracy:             {teacher_accuracy * 100:.1f}%")
print(f"Student (normal):             {normal_accuracy * 100:.1f}%")
print(f"Student (knowledge distilled): {kd_accuracy * 100:.1f}%")

print("\nSample predictions:\n")

for question, answer, features, label in test[:5]:

    prediction = (
        features.reshape(1, -1) @ kd_W + kd_b
    ).argmax(axis=1)[0]

    words = question.replace("?", "").split()

    a = int(words[-3])
    b = int(words[-1])

    odd = a if prediction == 0 else b
    even = b if prediction == 0 else a

    print("Input: ", question)
    print("Output:", f"{odd} is odd. {even} is even.")
    print()