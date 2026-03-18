d4(o4).
d6(o3).
d4(o1).
d2(o1).
d1(o3).
d3(o2).

d5(X) :- not d0(X), d4(X).
d8(X) :- not d3(X), d7(X).
d0(X) :- d9(X), not d5(X).
d6(X) :- not d7(X), d5(X).
d2(X) :- not d8(X), d1(X).
d8(X) :- not d3(X), d5(X).
d5(X) :- d8(X), not d4(X).
d0(X) :- not d5(X), d8(X).
d5(X) :- d2(X), not d6(X).
d7(X) :- not d5(X), d8(X).
d9(X) :- not d5(X), d8(X).