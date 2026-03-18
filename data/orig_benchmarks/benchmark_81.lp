d2(o2).
d3(o4).
d3(o8).
d4(o3).
d1(o2).

d1(X) :- d2(X), not d3(X), not d4(X).
d1(X) :- d0(X), not d5(X), not d3(X).
d5(X) :- d0(X), not d3(X), not d2(X).
d4(X) :- not d1(X), d0(X), not d3(X).
d5(X) :- not d0(X), d1(X), not d4(X).
d4(X) :- d0(X), not d3(X), not d2(X).
d1(X) :- d2(X), not d3(X), not d0(X).
d5(X) :- not d1(X), d0(X), not d2(X).
d1(X) :- d5(X), not d4(X), not d2(X).
d0(X) :- not d3(X), d5(X), not d2(X).
d1(X) :- not d0(X), d5(X), not d4(X).