d2(o7).
d3(o3).
d1(o8).
d4(o1).
d1(o6).
d3(o9).
d1(o3).
d0(o3).
d4(o3).

d3(X) :- not d5(X), d1(X), d4(X).
d0(X) :- d3(X), d1(X), not d4(X).
d5(X) :- d0(X), not d1(X), d4(X).
d0(X) :- not d2(X), d5(X), d1(X).
d1(X) :- d2(X), d3(X), not d0(X).
d3(X) :- d0(X), d5(X), not d4(X).
d0(X) :- d2(X), d5(X), not d4(X).
d1(X) :- d0(X), d5(X), not d4(X).
d4(X) :- not d2(X), d0(X), d5(X).
d5(X) :- not d2(X), d0(X), d4(X).
d1(X) :- d0(X), not d5(X), d4(X).