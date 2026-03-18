d8(o2).
d7(o8).
d8(o7).
d5(o9).
d5(o1).
d0(o1).
d0(o2).
d7(o3).
d1(o3).

d6(X) :- d0(X), not d7(X), d5(X).
d0(X) :- d3(X), not d7(X), d4(X).
d8(X) :- d5(X), not d0(X), d4(X).
d3(X) :- d0(X), not d7(X), d1(X).
d6(X) :- d0(X), not d1(X), d8(X).
d1(X) :- d9(X), d3(X), not d2(X).
d4(X) :- d2(X), not d1(X), d8(X).
d9(X) :- not d5(X), d7(X), d1(X).
d2(X) :- d5(X), d1(X), not d0(X).
d8(X) :- not d3(X), d2(X), d5(X).
d5(X) :- d9(X), not d7(X), d1(X).