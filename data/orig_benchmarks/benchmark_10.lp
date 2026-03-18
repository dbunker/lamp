d2(o3).
d9(o4).
d1(o5).
d0(o4).
d9(o1).
d5(o1).

d4(X) :- not d3(X), not d8(X), d5(X).
d8(X) :- d2(X), not d7(X), not d4(X).
d7(X) :- d5(X), not d4(X), not d9(X).
d0(X) :- not d3(X), not d7(X), d1(X).
d3(X) :- d6(X), not d7(X), not d0(X).
d1(X) :- not d6(X), not d7(X), d5(X).
d2(X) :- not d6(X), d8(X), not d9(X).
d1(X) :- not d8(X), not d7(X), d4(X).
d5(X) :- not d3(X), d7(X), not d0(X).
d3(X) :- d6(X), not d1(X), not d0(X).
d6(X) :- d3(X), not d7(X), not d5(X).