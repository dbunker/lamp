d3(o2).
d5(o2).
d4(o7).
d5(o7).
d4(o6).

d5(X) :- d2(X), d0(X), not d4(X).
d3(X) :- d4(X), d5(X), not d1(X).
d0(X) :- d2(X), d5(X), not d4(X).
d3(X) :- d2(X), not d5(X), d4(X).
d2(X) :- not d1(X), d0(X), d3(X).
d0(X) :- d4(X), not d5(X), d3(X).
d4(X) :- d2(X), d0(X), not d1(X).
d5(X) :- d1(X), d3(X), not d4(X).
d3(X) :- d4(X), d1(X), not d2(X).
d4(X) :- d5(X), d3(X), not d2(X).
d1(X) :- d4(X), d5(X), not d0(X).