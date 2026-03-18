d2(o3).
d0(o7).
d9(o8).
d8(o0).
d3(o4).
d9(o7).
d4(o2).
d2(o2).
d7(o7).

d1(X) :- d3(X), not d7(X), not d4(X).
d4(X) :- d2(X), not d5(X), not d0(X).
d5(X) :- not d3(X), d2(X), not d7(X).
d6(X) :- not d2(X), not d3(X), d1(X).
d7(X) :- not d2(X), d6(X), not d0(X).
d9(X) :- not d2(X), d0(X), not d3(X).
d1(X) :- d9(X), not d7(X), not d4(X).
d4(X) :- not d8(X), d6(X), not d1(X).
d9(X) :- not d3(X), not d8(X), d5(X).
d2(X) :- not d8(X), d7(X), not d9(X).
d0(X) :- not d8(X), d1(X), not d5(X).