d8(o2).
d7(o0).
d9(o5).
d4(o0).
d5(o4).
d6(o2).

d1(X) :- d9(X), not d8(X).
d9(X) :- not d4(X), d8(X).
d7(X) :- not d4(X), d8(X).
d8(X) :- d5(X).
d6(X) :- d4(X).