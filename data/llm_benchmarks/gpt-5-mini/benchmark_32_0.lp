d9(o5).
d5(o5).
d8(o3).
d7(o2).
d6(o5).
d5(o3).

d4(X) :- d5(X), d7(X).
d0(X) :- d6(X), not d9(X).
d6(X) :- d5(X).
d7(X) :- d8(X).