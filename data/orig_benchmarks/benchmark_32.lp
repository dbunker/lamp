d9(o5).
d5(o5).
d8(o3).
d7(o2).
d6(o5).
d5(o3).

d1(X) :- d3(X), d5(X), not d0(X).
d4(X) :- d0(X), d7(X), not d9(X).
d6(X) :- not d3(X), d5(X), d8(X).
d0(X) :- not d2(X), d6(X), d7(X).
d0(X) :- not d2(X), d1(X), d4(X).
d3(X) :- d2(X), d5(X), not d6(X).
d2(X) :- d3(X), d5(X), not d4(X).
d4(X) :- d9(X), not d8(X), d7(X).
d5(X) :- d0(X), d8(X), not d3(X).
d7(X) :- d6(X), not d1(X), d8(X).
d7(X) :- not d2(X), d3(X), d6(X).