d2(o2).
d2(o0).
d5(o4).
d9(o1).
d4(o4).
d2(o4).
d8(o0).
d6(o4).
d8(o4).

d5(X) :- not d8(X), not d9(X), d9(Y), d2(X).
d5(X) :- d9(X).
d2(X) :- d9(X).
d7(X) :- d8(X), not d6(X).
d3(X) :- not d4(X), d8(X), d2(X).