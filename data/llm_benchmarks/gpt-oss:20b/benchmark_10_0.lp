d2(o3).
d9(o4).
d1(o5).
d0(o4).
d9(o1).
d5(o1).

d8(X) :- not d9(X), d2(X).
d0(X) :- not d9(X), d1(X).
d0(X) :- d5(X), not d2(X).
d4(X) :- d5(X), not d6(X).
d1(X) :- d5(X), not d3(X).